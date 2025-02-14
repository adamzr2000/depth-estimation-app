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

    # Normalize the depth map for visualization
    depth_map_normalized = (depth_map - np.min(depth_map)) / (np.max(depth_map) - np.min(depth_map))
    depth_map_normalized = (depth_map_normalized * 255).astype(np.uint8)

    # Apply a colormap for better visualization
    depth_colormap = cv2.applyColorMap(depth_map_normalized, cv2.COLORMAP_JET)

    # Display the original frame and the depth map
    cv2.imshow("Original Frame", frame)
    cv2.imshow("Depth Map", depth_colormap)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
