from flask import Flask, render_template, Response, jsonify
import os
import cv2
import numpy as np
import time
from queue import Queue
from threading import Thread
import tensorflow as tf
import keras
import logging
import argparse

# -------------------------------------------------------------------------
# Argument parsing: choose execution mode and test flag
# -------------------------------------------------------------------------
parser = argparse.ArgumentParser(
    description="Run depth estimation in default (Keras) or deploy (SOL) mode.")
parser.add_argument(
    "-m", "--mode",
    choices=["default", "deploy"],
    default="default",
    help="Mode: 'default' uses Keras; 'deploy' uses SOL deployment.")
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

# -------------------------------------------------------------------------
# Suppress verbose TensorFlow logs
# -------------------------------------------------------------------------
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# -------------------------------------------------------------------------
# Logging configuration
# -------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
depth_logger = logging.getLogger("depth_estimation")
werkzeug_logger = logging.getLogger("werkzeug")
werkzeug_logger.setLevel(logging.ERROR)
depth_logger.info(f"Selected mode: {mode}")
depth_logger.info(f"Test mode: {'enabled' if use_test else 'disabled'}")

# -------------------------------------------------------------------------
# Log framework versions
# -------------------------------------------------------------------------
depth_logger.info(f"Keras version: {keras.__version__}")
depth_logger.info(f"TensorFlow version: {tf.__version__}")

# -------------------------------------------------------------------------
# If in deploy mode, prevent TF from grabbing GPUs
# -------------------------------------------------------------------------
if mode == "deploy":
    try:
        tf.config.set_visible_devices([], 'GPU')
        depth_logger.info("Hid TensorFlow GPUs in deploy mode to avoid context conflicts.")
    except Exception as e:
        depth_logger.warning(f"Could not hide GPUs: {e}")
elif use_gpu:
    depth_logger.warning("Ignoring --gpu flag since mode is not 'deploy'.")

# -------------------------------------------------------------------------
# Device summary
# -------------------------------------------------------------------------
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    depth_logger.info(f"GPUs Detected: {len(gpus)}")
    for idx, gpu in enumerate(gpus):
        depth_logger.info(f"  GPU {idx}: {gpu}")
else:
    depth_logger.info("No GPUs Detected (TensorFlow)")

# -------------------------------------------------------------------------
# Load model (Keras or SOL deployment)
# -------------------------------------------------------------------------
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models')
initialized = False

if mode == "deploy":
    deploy_folder = "monocular_deployed_gpu" if use_gpu else "monocular_deployed"
    depth_logger.info(f"Using SOL-optimized model from: {deploy_folder}")
    if use_gpu:
        from models.monocular_deployed_gpu.sol_monocular_example import sol_monocular
    else:
        from models.monocular_deployed.sol_monocular_example import sol_monocular
    deploy_path = os.path.join(MODEL_DIR, deploy_folder)
    mod = sol_monocular(deploy_path)
    model = None
else:
    depth_logger.info("Loading local Keras model...")
    keras_path = os.path.join(MODEL_DIR, 'monocular_keras')
    model = tf.keras.models.load_model(keras_path)
    depth_logger.info("Keras model loaded from %s", keras_path)

# Decide which model string and GPU flag for metrics
model_name = "monocular_sol" if mode == "deploy" else "monocular_keras"
gpu_flag   = bool(use_gpu) if mode == "deploy" else False

# -------------------------------------------------------------------------
# Video capture setup
# -------------------------------------------------------------------------
source = 0
if use_test:
    test_video_path = os.path.join(os.path.dirname(__file__),
                                   'videos', 'test-video.mp4')
    depth_logger.info(f"Opening test video: {test_video_path}")
    source = test_video_path

cap = cv2.VideoCapture(source)
if not cap.isOpened():
    depth_logger.error(f"Cannot open source: {source}")
    exit(1)

# Read source FPS and compute interval for pacing
raw_fps = cap.get(cv2.CAP_PROP_FPS)
source_fps = raw_fps if raw_fps and raw_fps > 1.0 else 30.0
frame_interval = 1.0 / source_fps

# -------------------------------------------------------------------------
# Background capture thread (always keeps latest frame in queue)
# -------------------------------------------------------------------------
frame_queue = Queue(maxsize=1)

def capture_loop():
    while True:
        ret, frame = cap.read()
        if not ret:
            if use_test:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            else:
                break
        if frame_queue.full():
            try:
                frame_queue.get_nowait()
            except:
                pass
        frame_queue.put(frame)

Thread(target=capture_loop, daemon=True).start()

# -------------------------------------------------------------------------
# Flask setup
# -------------------------------------------------------------------------
app = Flask(__name__)
WIDTH, HEIGHT = 512, 512

# Metrics counters
frame_count = total_inference_time = inference_count = 0
last_latency = last_fps = last_avg_latency = 0.0

# -------------------------------------------------------------------------
# Frame preprocessing
# -------------------------------------------------------------------------
def preprocess_frame(frame):
    resized = cv2.resize(frame, (WIDTH, HEIGHT))
    normalized = resized.astype('float32') / 255.0
    return np.expand_dims(normalized, axis=0)

# -------------------------------------------------------------------------
# Generators
# -------------------------------------------------------------------------
def generate_original_frames():
    last_emit = time.monotonic()
    while True:
        frame = frame_queue.get()
        # encode with 80% JPEG quality
        _, buf = cv2.imencode('.jpg', frame,
                              [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' +
               buf.tobytes() + b'\r\n')
        # throttle to video FPS
        now = time.monotonic()
        elapsed = now - last_emit
        to_sleep = frame_interval - elapsed
        if to_sleep > 0:
            time.sleep(to_sleep)
        last_emit = time.monotonic()

def generate_depth_frames():
    global model, initialized
    global frame_count, total_inference_time, inference_count
    global last_latency, last_fps, last_avg_latency

    if mode == "deploy" and not initialized:
        mod.init()
        initialized = True
        depth_logger.info("SOL deployment initialized in worker thread.")

    once = False
    while True:
        frame = frame_queue.get()
        inp = preprocess_frame(frame)

        if mode == "deploy" and not once:
            depth_logger.info("Setting I/O and optimizing deploy module...")
            mod.set_IO(inp)
            mod.optimize(2)
            class ModelWrapper:
                @staticmethod
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

        _, buf = cv2.imencode('.jpg', depth_col,
                              [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' +
               buf.tobytes() + b'\r\n')

# -------------------------------------------------------------------------
# Metrics endpoint
# -------------------------------------------------------------------------
@app.route('/metrics')
def metrics():
    avg_fps = inference_count / total_inference_time if total_inference_time > 0 else 0
    return jsonify(
        latency_ms      = round(last_latency    * 1000, 2),
        fps             = round(last_fps,         2),
        avg_latency_ms  = round(last_avg_latency * 1000, 2),
        avg_fps         = round(avg_fps,          2),
        model           = model_name,
        gpu_enabled     = gpu_flag
    )

# -------------------------------------------------------------------------
# Flask routes
# -------------------------------------------------------------------------
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

# -------------------------------------------------------------------------
# Entry point
# -------------------------------------------------------------------------
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5554)

