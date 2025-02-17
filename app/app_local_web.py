from flask import Flask, render_template, Response
import cv2
import numpy as np
import os
from huggingface_hub import from_pretrained_keras
import tensorflow as tf
import keras

app = Flask(__name__)

# Suppress verbose TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["KERAS_BACKEND"] = "tensorflow"

print(f"Keras version: {keras.__version__}")
print(f"TensorFlow Version: {tf.__version__}")

# Check if TensorFlow was built with GPU support
gpu_support = tf.test.is_built_with_cuda()
print(f"Built with GPU Support: {'Yes' if gpu_support else 'No'}")

# List physical GPUs detected by TensorFlow
gpus = tf.config.list_physical_devices('GPU')

if gpus:
    print(f"GPUs Detected: {len(gpus)}")
    for idx, gpu in enumerate(gpus):
        print(f"  GPU {idx}: {gpu}")
else:
    print("No GPUs Detected")
    # Print CPU details
    cpu_devices = tf.config.list_physical_devices('CPU')
    print(f"CPUs Detected: {len(cpu_devices)}")
    for idx, cpu in enumerate(cpu_devices):
        print(f"  CPU {idx}: {cpu}")

# Load the pretrained model from Hugging Face Hub
print("Loading pretrained model...")
model = from_pretrained_keras("keras-io/monocular-depth-estimation")
print("Model loaded successfully!")

# Initialize webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Error: Could not open video source.")

# Define input dimensions
HEIGHT, WIDTH = 256, 256

def preprocess_frame(frame):
    """Resize and normalize the frame for model input."""
    resized_frame = cv2.resize(frame, (WIDTH, HEIGHT))
    normalized_frame = resized_frame.astype("float32") / 255.0
    input_frame = np.expand_dims(normalized_frame, axis=0)
    return input_frame

def generate_original_frames():
    """Generate original video frames."""
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        _, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

def generate_depth_frames():
    """Generate depth estimation frames."""
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        input_frame = preprocess_frame(frame)
        depth_map = model.predict(input_frame)[0, :, :, 0]
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
