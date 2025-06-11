import os
from time import time
import argparse
import numpy as np
import cv2
import tensorflow as tf
import keras

argParser = argparse.ArgumentParser()
argParser.add_argument("-m", "--mode", choices=["default", "local", "sol", "deploy"], default="default")
argParser.add_argument(
    "--gpu",
    action="store_true",
    help="Use GPU-optimized SOL deployment (only valid with -m deploy)")

args = argParser.parse_args()

use_local = args.mode == "local"
use_sol = args.mode == "sol"
use_deploy = args.mode == "deploy"
use_deploy_gpu = args.gpu

if use_sol:
    import sol

if not use_deploy and use_deploy_gpu:
    print("Ignoring --gpu flag since mode is not 'deploy'.")

# Suppress TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# os.environ["KERAS_BACKEND"] = "tensorflow"

# Config
WIDTH, HEIGHT = 512, 512
# WIDTH, HEIGHT = 1024, 768
INPUT_DIR = os.path.join("images", "input")
OUTPUT_DIR = os.path.join("images", "output")

# Print environment info
print("=== TensorFlow Environment Test ===")
print(f"Keras version: {keras.__version__}")
print(f"TensorFlow version: {tf.__version__}")
print(f"Built with GPU support: {'Yes' if tf.test.is_built_with_cuda() else 'No'}")

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print("GPUs detected:")
    for gpu in gpus:
        print("  ", gpu)
else:
    print("No GPU detected.")


# Load pretrained depth model
print("Loading pretrained model...")
if use_sol or use_local:
    with tf.device('/cpu:0'):
        model = tf.keras.models.load_model("monocular")
elif use_deploy:
    deploy_folder = "monocular_deployed_gpu" if use_deploy_gpu else "monocular_deployed"
    print(f"Initializing SOL-optimized model from: {deploy_folder}")
    if use_deploy_gpu:
        from models.monocular_deployed_gpu.sol_monocular_example import sol_monocular
    else:
        from models.monocular_deployed.sol_monocular_example import sol_monocular
    deploy_path = os.path.join("models", deploy_folder)
    mod = sol_monocular(deploy_path)
    mod.init()
    vdims = np.ndarray((1), dtype=np.int64)
else:
    import huggingface_hub
    model = huggingface_hub.from_pretrained_keras("keras-io/monocular-depth-estimation")

print("Model loaded successfully!")


# Ensure output folder exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

def preprocess_frame(frame):
    # Resize the input frame to the target dimensions (WIDTH, HEIGHT)
    resized_frame = cv2.resize(frame, (WIDTH, HEIGHT))
    # Convert the resized frame to float32 type and normalize pixel values to [0, 1]
    normalized_frame = resized_frame.astype("float32") / 255.0
    # Add a batch dimension to the frame (shape becomes (1, HEIGHT, WIDTH, 3))
    return np.expand_dims(normalized_frame, axis=0)


def generate_depth_frames(input_dir=INPUT_DIR, width=WIDTH, height=HEIGHT):
    """
    Generator that yields (filename, depth_colored) for each image in the input directory.
    """
    global model
    once = False
    image_files = sorted([
        f for f in os.listdir(input_dir)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ])

    if not image_files:
        print(f"No images found in '{input_dir}'")
        return
    t0 = time()
    for i, filename in enumerate(image_files, start=1):
        t_before = time()
        path = os.path.join(input_dir, filename)
        image = cv2.imread(path)
        if image is None:
            print(f"Could not read image: {filename}")
            continue


        print(f"[{i}/{len(image_files)}] Processing: {filename}")
        input_tensor = preprocess_frame(image)
        if not once:
            once = True
            if use_sol:
                print("Optimizing")
                # model.predict(input_tensor)
                model = sol.optimize(model, [input_tensor], vdims=[False])
            if use_deploy:
                mod.set_IO(input_tensor)
                mod.optimize(2)
                class model:
                    def predict(input):
                        return mod(input)
        
        print("Before:", time()-t_before)
        t_model = time()
        depth_map = model.predict(input_tensor)[0, :, :, 0]
        print("Model:", time()-t_model)

        # Normalize and apply colormap
        t_after = time()
        normalized = (depth_map - np.min(depth_map)) / (np.max(depth_map) - np.min(depth_map) + 1e-8)
        depth_colored = cv2.applyColorMap((normalized * 255).astype(np.uint8), cv2.COLORMAP_JET)
        print("After:", time()-t_after)
        yield filename, depth_colored
    t1 = time()
    print("Total:", t1-t0)

# Run full batch processing
if __name__ == "__main__":
    for filename, depth_colored in generate_depth_frames():
        base = os.path.splitext(filename)[0]
        sol_string = "sol_" if use_sol else "deploy_" if use_deploy else "local_" if use_local else ""
        output_path = os.path.join(OUTPUT_DIR, f"{base}_{sol_string}depth.png")
        cv2.imwrite(output_path, depth_colored)
        print(f"  → Saved: {output_path}")

    print("✅ All images processed and saved.")
