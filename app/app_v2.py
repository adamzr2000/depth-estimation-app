from flask import Flask, render_template, Response
import cv2
import numpy as np
import tensorflow as tf
from huggingface_hub import from_pretrained_keras

# Initialize Flask app
app = Flask(__name__)

# Load the pretrained model from Hugging Face Hub
print("Loading pretrained model...")
model = from_pretrained_keras("keras-io/monocular-depth-estimation")
print("Model loaded successfully!")

# Initialize webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open video source.")
    exit()

# Define input dimensions
HEIGHT, WIDTH = 256, 256

def preprocess_frame(frame):
    """Resize and normalize the frame for model input."""
    resized_frame = cv2.resize(frame, (WIDTH, HEIGHT))  # Resize to model input size
    normalized_frame = tf.image.convert_image_dtype(resized_frame, tf.float32)  # Normalize to [0, 1]
    input_frame = np.expand_dims(normalized_frame, axis=0)  # Add batch dimension
    return input_frame

def generate_frames():
    """Generator function to yield video frames."""
    while True:
        ret, frame = cap.read()
        if not ret:
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

        # Resize depth_colormap to match the original frame size
        depth_colormap_resized = cv2.resize(depth_colormap, (frame.shape[1], frame.shape[0]))

        # Combine original and depth map for display
        combined_frame = np.hstack((frame, depth_colormap_resized))

        # Encode frame to JPEG
        _, buffer = cv2.imencode('.jpg', combined_frame)
        frame_data = buffer.tobytes()

        # Yield the frame data as part of an HTTP response
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_data + b'\r\n')

@app.route('/')
def index():
    """Render the main page."""
    return render_template('index_v2.html')

@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
