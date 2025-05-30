from flask import Flask, render_template, Response, jsonify
import os
import cv2
import numpy as np
import time
import tensorflow as tf
import keras
import logging
import argparse
import shutil

# -----------------------------------------------------------------------------
# Argument parsing: choose execution mode and test flag
# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser(
    description="Run depth estimation in default (Keras), deploy, or test video mode.")
parser.add_argument(
    "-m", "--mode",
    choices=["default", "deploy"],
    default="default",
    help="Mode: 'default' uses HuggingFace/Keras; 'deploy' uses sol_monocular deployment.")
parser.add_argument(
    "-t", "--test",
    action="store_true",
    help="Test mode: read looped video from 'videos/test-video.mp4' instead of webcam.")
args = parser.parse_args()
mode = args.mode
use_test = args.test

# -----------------------------------------------------------------------------
# Suppress TensorFlow and Keras verbose logs
# -----------------------------------------------------------------------------
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['KERAS_BACKEND'] = 'tensorflow'

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
# Log framework versions & hardware
# -----------------------------------------------------------------------------
depth_logger.info(f"Keras version: {keras.__version__}")
depth_logger.info(f"TensorFlow version: {tf.__version__}")

gpu_support = tf.test.is_built_with_cuda()
depth_logger.info(f"Built with GPU support: {'Yes' if gpu_support else 'No'}")

# -----------------------------------------------------------------------------
# Load and initialize model based on mode
# -----------------------------------------------------------------------------
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models')
if mode == "deploy":
    depth_logger.info("Initializing SOL-optimized model from local files...")
    from models.monocular_deployed.sol_monocular_example import sol_monocular
    deploy_path = os.path.join(MODEL_DIR, 'monocular_deployed')
    mod = sol_monocular(deploy_path)
    mod.init()
    vdims = np.ndarray((1,), dtype=np.int64)

    class DeployedModel:
        def __init__(self, mod, vdims):
            self.mod = mod
            self.vdims = vdims

        def predict(self, input_tensor):
            out = np.zeros((1, 256, 256, 1), dtype=np.float32)
            args = [input_tensor, out, self.vdims]
            self.mod.set_IO(args)
            self.mod.run()
            return out

    model = DeployedModel(mod, vdims)
    depth_logger.info("Deployed model initialized successfully!")
else:
    depth_logger.info("Loading local Keras model...")
    keras_path = os.path.join(MODEL_DIR, 'monocular_keras')
    model = tf.keras.models.load_model(keras_path)
    depth_logger.info("Keras model loaded from %s", keras_path)

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

