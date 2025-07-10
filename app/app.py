from flask import Flask, render_template, Response, jsonify
import os
import cv2
import numpy as np
import time
import tensorflow as tf
import keras
import logging
import argparse
import threading
import atexit

# -----------------------------------------------------------------------------
# Argument parsing
# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Run depth estimation in default (Keras), sol-optimize, deploy, or test video mode.")
parser.add_argument("-m", "--mode", choices=["default", "deploy"], default="default")
parser.add_argument("-t", "--test", action="store_true")
parser.add_argument("--gpu", action="store_true")
parser.add_argument("--vaccel", action="store_true")
args = parser.parse_args()

mode = args.mode
use_test = args.test
use_gpu = args.gpu
use_deploy_vaccel = args.vaccel

# -----------------------------------------------------------------------------
# Logging configuration
# -----------------------------------------------------------------------------
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
depth_logger = logging.getLogger("depth_estimation")
werkzeug_logger = logging.getLogger("werkzeug")
werkzeug_logger.setLevel(logging.ERROR)

depth_logger.info(f"Selected mode: {mode}")
depth_logger.info(f"Test mode: {'enabled' if use_test else 'disabled'}")
depth_logger.info(f"Keras version: {keras.__version__}")
depth_logger.info(f"TensorFlow version: {tf.__version__}")

# -----------------------------------------------------------------------------
# GPU Configuration
# -----------------------------------------------------------------------------
if mode == "deploy":
    try:
        tf.config.set_visible_devices([], 'GPU')
        depth_logger.info("Hid TensorFlow GPUs in deploy mode.")
    except Exception as e:
        depth_logger.warning(f"Could not hide GPUs: {e}")
elif use_gpu:
    depth_logger.warning("Ignoring --gpu flag since mode is not 'deploy'.")

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for idx, gpu in enumerate(gpus):
        depth_logger.info(f"GPU {idx}: {gpu}")
else:
    depth_logger.info("No GPUs Detected")

# -----------------------------------------------------------------------------
# Load model
# -----------------------------------------------------------------------------
initialized = False

if mode == "deploy":
    if use_deploy_vaccel:
        deploy_folder = ["monocular_deployed_vaccel"]
    elif use_gpu:
        deploy_folder = ["monocular_deployed", "lib_gpu"]
    else:
        deploy_folder = ["monocular_deployed", "lib_cpu"]

    deploy_path = os.path.join("models", *deploy_folder)

    if use_deploy_vaccel:
        if use_gpu:
            from models.monocular_deployed_vaccel.sol_monocular_vaccel import sol_monocular_gpu as sol_monocular
        else:
            from models.monocular_deployed_vaccel.sol_monocular_vaccel import sol_monocular
    elif use_gpu:
        from models.monocular_deployed.lib_gpu.sol_monocular_example import sol_monocular
    else:
        from models.monocular_deployed.lib_cpu.sol_monocular_example import sol_monocular

    depth_logger.info(f"Using SOL-optimized model from: {deploy_path}")
    mod = sol_monocular(deploy_path)
    vdims = np.ndarray((1,), dtype=np.int64)
    model = None
else:
    keras_path = os.path.join("models", "monocular_keras")
    model = tf.keras.models.load_model(keras_path)
    depth_logger.info(f"Keras model loaded from {keras_path}")

# -----------------------------------------------------------------------------
# Source setup
# -----------------------------------------------------------------------------
source = os.path.join(os.path.dirname(__file__), 'videos', 'test-video.mp4') if use_test else 0
cap = None
cap_lock = threading.Lock()

if not use_test:
    cap = cv2.VideoCapture(source)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 15)
    if cap.get(cv2.CAP_PROP_FOURCC) != 0.0:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

@atexit.register
def release_camera():
    if cap and cap.isOpened():
        cap.release()
        depth_logger.info("Released shared camera.")

# -----------------------------------------------------------------------------
# Flask setup
# -----------------------------------------------------------------------------
app = Flask(__name__)
WIDTH, HEIGHT = 512, 512

frame_count = total_inference_time = inference_count = 0
last_latency = last_fps = last_avg_latency = 0.0
model_name = "monocular_sol" if mode == "deploy" else "monocular_keras"
gpu_flag = bool(use_gpu) if mode == "deploy" else False

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
    if use_test:
        cap_local = cv2.VideoCapture(source)
        cap_local.set(cv2.CAP_PROP_FPS, 15)
        fps = cap_local.get(cv2.CAP_PROP_FPS)
        frame_interval = 1 / fps if fps > 0 else 1 / 30.0
        while True:
            start_time = time.time()
            ret, frame = cap_local.read()
            if not ret:
                cap_local.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            _, buf = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n')
            time.sleep(max(0, frame_interval - (time.time() - start_time)))
    else:
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = 1 / fps if fps > 0 else 1 / 30.0
        while True:
            start_time = time.time()
            with cap_lock:
                ret, frame = cap.read()
            if not ret:
                time.sleep(0.01)
                continue
            _, buf = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n')
            time.sleep(max(0, frame_interval - (time.time() - start_time)))

def generate_depth_frames():
    global model, initialized, frame_count, total_inference_time, inference_count
    global last_latency, last_fps, last_avg_latency

    if use_test:
        cap_local = cv2.VideoCapture(source)
        cap_local.set(cv2.CAP_PROP_FPS, 15)
        while True:
            ret, frame = cap_local.read()
            if not ret:
                cap_local.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            yield from process_depth_frame(frame)
    else:
        while True:
            with cap_lock:
                ret, frame = cap.read()
            if not ret:
                time.sleep(0.01)
                continue
            yield from process_depth_frame(frame)

def process_depth_frame(frame):
    global model, initialized, frame_count, total_inference_time, inference_count
    global last_latency, last_fps, last_avg_latency

    inp = preprocess_frame(frame)

    if mode == "deploy" and not initialized:
        mod.init()
        initialized = True
        depth_logger.info("SOL deployment initialized.")

    if mode == "deploy" and isinstance(model, type(None)):
        if use_gpu:
            mod.set_IO(inp)
            mod.optimize(2)
        class ModelWrapper:
            def predict(x):
                return mod(x)
        model = ModelWrapper

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
    _, buf = cv2.imencode('.jpg', depth_col, [int(cv2.IMWRITE_JPEG_QUALITY), 80])

    yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n')

# -----------------------------------------------------------------------------
# Metrics endpoint
# -----------------------------------------------------------------------------
@app.route('/metrics')
def metrics():
    avg_fps = inference_count / total_inference_time if total_inference_time > 0 else 0
    return jsonify(
        latency_ms=round(last_latency * 1000, 2),
        fps=round(last_fps, 2),
        avg_latency_ms=round(last_avg_latency * 1000, 2),
        avg_fps=round(avg_fps, 2),
        model=model_name,
        gpu_enabled=gpu_flag
    )

# -----------------------------------------------------------------------------
# Flask routes
# -----------------------------------------------------------------------------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/original_video_feed')
def original_video_feed():
    return Response(generate_original_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/depth_video_feed')
def depth_video_feed():
    return Response(generate_depth_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5554)
