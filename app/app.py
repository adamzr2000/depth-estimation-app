from flask import Flask, render_template, Response, jsonify
import os
import cv2
import numpy as np
import time
import tensorflow as tf
import keras
import logging
import argparse

# -----------------------------------------------------------------------------
# Argument parsing: choose execution mode and test flag
# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser(
    description="Run depth estimation in default (Keras), sol-optimize, deploy, or test video mode.")
parser.add_argument(
    "-m", "--mode",
    choices=["default", "sol", "deploy"],
    default="default",
    help="Mode: 'default' uses HuggingFace/Keras; 'sol' optimizes the Keras model with SOL; 'deploy' uses SOL deployment.")
parser.add_argument(
    "-t", "--test",
    action="store_true",
    help="Test mode: read looped video from 'videos/test-video.mp4' instead of webcam.")
parser.add_argument(
    "--gpu",
    action="store_true",
    help="Use GPU-optimized SOL deployment (only valid with -m deploy)")
args = parser.parse_args()
mode = args.mode
use_test = args.test
use_gpu = args.gpu
use_sol = (mode == "sol")

# -----------------------------------------------------------------------------
# Suppress verbose logs
# -----------------------------------------------------------------------------
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# -----------------------------------------------------------------------------
# Logging configuration
# -----------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
depth_logger = logging.getLogger("depth_estimation")
werkzeug_logger = logging.getLogger("werkzeug")
werkzeug_logger.setLevel(logging.ERROR)
depth_logger.info(f"Selected mode: {mode}")
depth_logger.info(f"Test mode: {'enabled' if use_test else 'disabled'}")

# -----------------------------------------------------------------------------
# Log framework versions
# -----------------------------------------------------------------------------
depth_logger.info(f"Keras version: {keras.__version__}")
depth_logger.info(f"TensorFlow version: {tf.__version__}")

# -----------------------------------------------------------------------------
# If in deploy mode, prevent TF from grabbing GPUs
# -----------------------------------------------------------------------------
if mode == "deploy":
    try:
        tf.config.set_visible_devices([], 'GPU')
        depth_logger.info("Hid TensorFlow GPUs in deploy mode to avoid context conflicts.")
    except Exception as e:
        depth_logger.warning(f"Could not hide GPUs: {e}")
elif use_gpu:
    depth_logger.warning("Ignoring --gpu flag since mode is not 'deploy'.")

# -----------------------------------------------------------------------------
# Device summary
# -----------------------------------------------------------------------------
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    depth_logger.info(f"GPUs Detected: {len(gpus)}")
    for idx, gpu in enumerate(gpus):
        depth_logger.info(f"  GPU {idx}: {gpu}")
else:
    depth_logger.info("No GPUs Detected (TensorFlow)")

# -----------------------------------------------------------------------------
# Load model skeleton; SOL init/deploy deferred into worker thread
# -----------------------------------------------------------------------------
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models')
initialized = False

if mode == "deploy":
    deploy_folder = "monocular_deployed_gpu" if use_gpu else "monocular_deployed"
    depth_logger.info(f"Using SOL-optimized model from: {deploy_folder}")
    # import SOL binding
    if use_gpu:
        from models.monocular_deployed_gpu.sol_monocular_example import sol_monocular
    else:
        from models.monocular_deployed.sol_monocular_example import sol_monocular
    deploy_path = os.path.join(MODEL_DIR, deploy_folder)
    mod = sol_monocular(deploy_path)
    vdims = np.ndarray((1,), dtype=np.int64)
    model = None

elif use_sol:
    depth_logger.info("Loading local Keras model for SOL optimization...")
    keras_path = os.path.join(MODEL_DIR, 'monocular_keras')
    with tf.device('/cpu:0'):
        model = tf.keras.models.load_model(keras_path)
    depth_logger.info("Keras model loaded for SOL.")

else:
    depth_logger.info("Loading local Keras model...")
    keras_path = os.path.join(MODEL_DIR, 'monocular_keras')
    model = tf.keras.models.load_model(keras_path)
    depth_logger.info("Keras model loaded from %s", keras_path)

# -----------------------------------------------------------------------------
# Video capture setup
# -----------------------------------------------------------------------------
source = 0
if use_test:
    test_video_path = os.path.join(os.path.dirname(__file__), 'videos', 'test-video.mp4')
    depth_logger.info(f"Opening test video: {test_video_path}")
    source = test_video_path

cap = cv2.VideoCapture(source)
if not cap.isOpened():
    depth_logger.error(f"Cannot open source: {source}")
    exit(1)

# -----------------------------------------------------------------------------
# Flask setup
# -----------------------------------------------------------------------------
app = Flask(__name__)
WIDTH, HEIGHT = 512, 512

# Metrics counters
frame_count = total_inference_time = inference_count = 0
last_latency = last_fps = last_avg_latency = 0.0

# -----------------------------------------------------------------------------
# Frame preprocessing
# -----------------------------------------------------------------------------
def preprocess_frame(frame):
    resized = cv2.resize(frame, (WIDTH, HEIGHT))
    normalized = resized.astype('float32') / 255.0
    return np.expand_dims(normalized, axis=0)

# -----------------------------------------------------------------------------
# Generators
# -----------------------------------------------------------------------------
def generate_original_frames():
    while True:
        ret, frame = cap.read()
        if not ret:
            if use_test:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            else:
                continue
        _, buf = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n')


def generate_depth_frames():
    global model, initialized, frame_count, total_inference_time, inference_count
    global last_latency, last_fps, last_avg_latency

    # Initialize SOL deployment if needed
    if mode == "deploy" and not initialized:
        mod.init()
        initialized = True
        depth_logger.info("SOL deployment initialized in worker thread.")

    once = False
    while True:
        ret, frame = cap.read()
        if not ret:
            if use_test:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            else:
                continue

        inp = preprocess_frame(frame)

        # SOL optimization on first frame
        if use_sol and not once:
            depth_logger.info("Optimizing Keras model with SOL...")
            model = sol.optimize(model, [inp], vdims=[False])
            once = True

        # Deploy optimization on first frame
        if mode == "deploy" and not once:
            depth_logger.info("Setting I/O and optimizing deploy module...")
            mod.set_IO(inp)
            mod.optimize(2)
            # wrap mod into predict API
            class ModelWrapper:
                
                def predict(x):
                    return mod(x)
            model = ModelWrapper
            once = True

        start = time.time()
        depth_map = model.predict(inp)[0, :, :, 0]
        latency = time.time() - start

        frame_count += 1
        total_inference_time += latency
        inference_count += 1
        last_latency = latency
        last_fps = 1.0 / latency if latency > 0 else float('inf')
        last_avg_latency = total_inference_time / inference_count

        mn, mx = float(depth_map.min()), float(depth_map.max())
        norm = (depth_map - mn) / (mx - mn + 1e-8)
        cmap = (norm * 255).astype(np.uint8)
        depth_col = cv2.applyColorMap(cmap, cv2.COLORMAP_JET)
        _, buf = cv2.imencode('.jpg', depth_col)

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n')

# -----------------------------------------------------------------------------
# Metrics endpoint
# -----------------------------------------------------------------------------
@app.route('/metrics')
def metrics():
    return jsonify(
        latency=round(last_latency, 4),
        fps=round(last_fps, 2),
        avg_latency=round(last_avg_latency, 4)
    )

# -----------------------------------------------------------------------------
# Flask routes
# -----------------------------------------------------------------------------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/original_video_feed')
def original_video_feed():
    return Response(generate_original_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/depth_video_feed')
def depth_video_feed():
    return Response(generate_depth_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5554)

