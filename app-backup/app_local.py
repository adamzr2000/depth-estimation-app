import cv2
import numpy as np
import tensorflow as tf
from huggingface_hub import from_pretrained_keras
import os
import keras

# Suppress verbose TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["KERAS_BACKEND"] = "tensorflow"

print("=== TensorFlow Environment Test ===")
print(f"Keras version: {keras.__version__}")

# Check TensorFlow version
print(f"TensorFlow Version: {tf.__version__}")

# Check if TensorFlow was built with GPU support
gpu_support = tf.test.is_built_with_cuda()
print(f"Built with GPU Support: {'Yes' if gpu_support else 'No'}")

# Load the pretrained model from Hugging Face Hub
print("Loading pretrained model...")
model = from_pretrained_keras("keras-io/monocular-depth-estimation")
print("Model loaded successfully!")

# Initialize webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open video source.")
    exit()

# Define input dimensions (based on the model's requirements)
HEIGHT, WIDTH = 256, 256

def preprocess_frame(frame):
    """Resize and normalize the frame for model input."""
    resized_frame = cv2.resize(frame, (WIDTH, HEIGHT))  # Resize to model input size
    normalized_frame = tf.image.convert_image_dtype(resized_frame, tf.float32)  # Normalize to [0, 1]
    input_frame = np.expand_dims(normalized_frame, axis=0)  # Add batch dimension
    return input_frame

frame_count = 0
while True:
    # Capture a frame from the webcam
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    # Preprocess the frame for prediction
    input_frame = preprocess_frame(frame)

    # Predict the depth map
    depth_map = model.predict(input_frame)[0, :, :, 0]

    # Extract depth statistics
    min_depth = np.min(depth_map)
    max_depth = np.max(depth_map)
    avg_depth = np.mean(depth_map)
    std_depth = np.std(depth_map)
    median_depth = np.median(depth_map)
    
    frame_count += 1
    print(f"Frame {frame_count}:")
    print(f"  - Min Depth: {min_depth:.4f} (Lowest estimated depth in the frame)")
    print(f"  - Max Depth: {max_depth:.4f} (Highest estimated depth in the frame)")
    print(f"  - Avg Depth: {avg_depth:.4f} (Mean depth across the entire frame)")
    print(f"  - Std Depth: {std_depth:.4f} (Standard deviation, indicating depth variation)")
    print(f"  - Median Depth: {median_depth:.4f} (Middle value of depth distribution)")
    print("---------------------------------------------------")

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Exit requested. Stopping...")
        break

# Release the webcam
cap.release()
print("Webcam released. Exiting.")
