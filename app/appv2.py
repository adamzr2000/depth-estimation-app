#!/usr/bin/env python3
import os
import argparse
import logging

# -----------------------------------------------------------------------------
# Argument parsing
# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser(
    description="Run depth estimation in default (Keras) or deploy (SOL) modes, with optional test video and GPU-offload.")
parser.add_argument(
    "-m", "--mode",
    choices=["default", "deploy"],
    default="default",
    help="Mode: 'default' uses HuggingFace/Keras; 'deploy' uses SOL deployment.")
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

# -----------------------------------------------------------------------------
# Logging configuration
# -----------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
depth_logger = logging.getLogger("depth_estimation")

# -----------------------------------------------------------------------------
# Warn invalid flag usage
# -----------------------------------------------------------------------------
if mode != "deploy" and use_gpu:
    depth_logger.warning("Ignoring --gpu flag since mode is not 'deploy'.")

# -----------------------------------------------------------------------------
# Suppress TensorFlow logs and Keras backend default
# -----------------------------------------------------------------------------
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['KERAS_BACKEND'] = 'tensorflow'

# -----------------------------------------------------------------------------
# Hide GPUs from TensorFlow when using SOL GPU
# -----------------------------------------------------------------------------
if mode == "deploy" and use_gpu:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    depth_logger.info("Hid GPUs from TensorFlow; SOL will initialize CUDA context")

# -----------------------------------------------------------------------------
# Initialize SOL-based model if deploying
# -----------------------------------------------------------------------------
import numpy as np
if mode == "deploy":
    deploy_folder = "monocular_deployed_gpu" if use_gpu else "monocular_deployed"
    depth_logger.info(f"Initializing SOL-optimized model from: {deploy_folder}")
    if use_gpu:
        from models.monocular_deployed_gpu.sol_monocular_example import sol_monocular
    else:
        from models.monocular_deployed.sol_monocular_example import sol_monocular

    MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models')
    deploy_path = os.path.join(MODEL_DIR, deploy_folder)
    sol_mod = sol_monocular(deploy_path)
    sol_mod.init()
    vdims = np.ndarray((1,), dtype=np.int64)

    class DeployedModel:
        def __init__(self, sol_lib, vdims):
            self.sol = sol_lib
            self.vdims = vdims

        def predict(self, input_tensor):
            # re-open and re-init to ensure valid CUDA context per call
            self.sol.open()
            self.sol.init()
            out = np.zeros((1, 256, 256, 1), dtype=np.float32)
            return self.sol.run(input_tensor, out, self.vdims)

    model = DeployedModel(sol_mod, vdims)
    depth_logger.info("SOL-deployed model initialized successfully!")("SOL-deployed model initialized successfully!")
else:
    # -------------------------------------------------------------------------
    # Load Keras model for default mode
    # -------------------------------------------------------------------------
    import tensorflow as tf
    import keras
    depth_logger.info(f"Keras version: {keras.__version__}")
    depth_logger.info(f"TensorFlow version: {tf.__version__}")
    keras_path = os.path.join(os.path.dirname(__file__), 'models', 'monocular_keras')
    model = tf.keras.models.load_model(keras_path)
    depth_logger.info(f"Keras model loaded from {keras_path}")


# Heavy imports for video and web
# -----------------------------------------------------------------------------
import cv2
import time
from flask import Flask, render_template, Response, jsonify

# -----------------------------------------------------------------------------
# Video capture setup: webcam or test video
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
# Flask application setup
# -----------------------------------------------------------------------------
app = Flask(__name__)

# Input dimensions for preprocessing
WIDTH, HEIGHT = 512, 512

# Metrics for inference performance
depth_logger.info("Starting video stream")
frame_count = 0
total_inference_time = 0.0
inference_count = 0
last_latency = 0.0
last_fps = 0.0
last_avg_latency = 0.0

# -----------------------------------------------------------------------------
# Frame preprocessing helper
# -----------------------------------------------------------------------------
def preprocess_frame(frame):
    resized = cv2.resize(frame, (WIDTH, HEIGHT))
    normalized = resized.astype('float32') / 255.0
    return np.expand_dims(normalized, axis=0)

# -----------------------------------------------------------------------------
# Generator: raw video frames
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

# -----------------------------------------------------------------------------
# Generator: depth‐estimated video frames
# -----------------------------------------------------------------------------
def generate_depth_frames():
    global frame_count, total_inference_time, inference_count, last_latency, last_fps, last_avg_latency
    while True:
        ret, frame = cap.read()
        if not ret:
            if use_test:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            else:
                continue

        inp = preprocess_frame(frame)
        start = time.time()
        depth_map = model.predict(inp)[0, :, :, 0]
        latency = time.time() - start
        frame_count += 1
        total_inference_time += latency
        inference_count += 1
        avg_latency = total_inference_time / inference_count
        fps = 1.0 / latency if latency > 0 else float('inf')
        # update globals for metrics endpoint
        last_latency = latency
        last_fps = fps
        last_avg_latency = avg_latency

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
# Flask routes: index & video feeds
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

