from flask import Flask, render_template, Response, request
import cv2
import numpy as np
import os
import threading
from huggingface_hub import from_pretrained_keras
import tensorflow as tf
import keras
import logging

app = Flask(__name__)

# Suppress verbose TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["KERAS_BACKEND"] = "tensorflow"

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
depth_logger = logging.getLogger("depth_estimation")
werkzeug_logger = logging.getLogger("werkzeug")
werkzeug_logger.setLevel(logging.ERROR)  # Suppress Flask's default request logs

depth_logger.info(f"Keras version: {keras.__version__}")
depth_logger.info(f"TensorFlow Version: {tf.__version__}")

gpu_support = tf.test.is_built_with_cuda()
depth_logger.info(f"Built with GPU Support: {'Yes' if gpu_support else 'No'}")

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    depth_logger.info(f"GPUs Detected: {len(gpus)}")
    for idx, gpu in enumerate(gpus):
        depth_logger.info(f"  GPU {idx}: {gpu}")
else:
    depth_logger.info("No GPUs Detected")
    cpu_devices = tf.config.list_physical_devices('CPU')
    depth_logger.info(f"CPUs Detected: {len(cpu_devices)}")
    for idx, cpu in enumerate(cpu_devices):
        depth_logger.info(f"  CPU {idx}: {cpu}")

depth_logger.info("Loading pretrained model...")
model = from_pretrained_keras("keras-io/monocular-depth-estimation")
depth_logger.info("Model loaded successfully!")

# Define input dimensions
# WIDTH, HEIGHT = 256, 256
WIDTH, HEIGHT = 512, 512

# Shared frame buffer and lock
frame_buffer = None
buffer_lock = threading.Lock()
frame_count = 0

def preprocess_frame(frame):
    # Resize the input frame to the target dimensions (WIDTH, HEIGHT)
    resized_frame = cv2.resize(frame, (WIDTH, HEIGHT))
    # Convert the resized frame to float32 type and normalize pixel values to [0, 1]
    normalized_frame = resized_frame.astype("float32") / 255.0
    # Add a batch dimension to the frame (shape becomes (1, HEIGHT, WIDTH, 3))
    return np.expand_dims(normalized_frame, axis=0)

@app.route('/upload_frame', methods=['POST'])
def upload_frame():
    global frame_buffer
    nparr = np.frombuffer(request.data, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None:
        return Response("Invalid frame", status=400)
    with buffer_lock:
        frame_buffer = frame
    return Response("Frame received", status=200)

def generate_original_frames():
    while True:
        with buffer_lock:
            if frame_buffer is None:
                continue
            frame = frame_buffer.copy()
        _, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

def generate_depth_frames():
    global frame_count
    while True:
        with buffer_lock:
            if frame_buffer is None:
                continue
            frame = frame_buffer.copy()
        input_frame = preprocess_frame(frame)
        depth_map = model.predict(input_frame)[0, :, :, 0]
        
        print("Input buffer shape:", input_frame.shape)
        print("Output buffer shape:", depth_map.shape)

        # Extract depth statistics
        min_depth = np.min(depth_map)
        max_depth = np.max(depth_map)
        avg_depth = np.mean(depth_map)
        std_depth = np.std(depth_map)
        median_depth = np.median(depth_map)
        
        frame_count += 1
        depth_logger.info(f"Frame {frame_count}:")
        depth_logger.info(f"  - Min Depth: {min_depth:.4f} (Lowest estimated depth in the frame)")
        depth_logger.info(f"  - Max Depth: {max_depth:.4f} (Highest estimated depth in the frame)")
        depth_logger.info(f"  - Avg Depth: {avg_depth:.4f} (Mean depth across the entire frame)")
        depth_logger.info(f"  - Std Depth: {std_depth:.4f} (Standard deviation, indicating depth variation)")
        depth_logger.info(f"  - Median Depth: {median_depth:.4f} (Middle value of depth distribution)")
        depth_logger.info("---------------------------------------------------")

        depth_map_normalized = (depth_map - np.min(depth_map)) / (np.max(depth_map) - np.min(depth_map))
        depth_map_normalized = (depth_map_normalized * 255).astype(np.uint8)
        depth_colormap = cv2.applyColorMap(depth_map_normalized, cv2.COLORMAP_JET)
        _, buffer = cv2.imencode('.jpg', depth_colormap)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/original_video_feed')
def original_video_feed():
    return Response(generate_original_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/depth_video_feed')
def depth_video_feed():
    return Response(generate_depth_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)