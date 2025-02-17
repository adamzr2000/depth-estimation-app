from flask import Flask, render_template, Response, request
import cv2
import numpy as np
import os
import threading
import logging
import ctypes
from numpy.ctypeslib import ndpointer, as_ctypes_type

app = Flask(__name__)

# Suppress verbose logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
depth_logger = logging.getLogger("depth_estimation")
werkzeug_logger = logging.getLogger("werkzeug")
werkzeug_logger.setLevel(logging.ERROR)  # Suppress Flask's default request logs

# Define input dimensions
HEIGHT, WIDTH = 256, 256

# Shared frame buffer and lock
frame_buffer = None
buffer_lock = threading.Lock()
frame_count = 0

# SOL Monocular Wrapper
class sol_monocular:
	def create(self):
		self.lib = ctypes.CDLL(self.path + "/" + "libsol_monocular.so")
		self.call = self.lib.__getattr__("sol_predict") 
		self.call.restype = None

		self.init = self.lib.__getattr__("sol_monocular_init")
		self.init.restype = None
		self.init.argtypes = None

		self.free = self.lib.__getattr__("sol_monocular_free")
		self.free.restype = None
		self.free.argtypes = None

		self.seed_ = self.lib.__getattr__("sol_monocular_set_seed")
		self.seed_.argtypes = [ctypes.c_uint64]
		self.seed_.restype = None

		self.set_IO_ = self.lib.__getattr__("sol_monocular_set_IO")
		self.set_IO_.restype = None

		self.call_no_args = self.lib.__getattr__("sol_monocular_run")
		self.call_no_args.argtypes = None
		self.call_no_args.restype = None

		self.get_output = self.lib.__getattr__("sol_monocular_get_output")
		self.get_output.argtypes = None
		self.get_output.restype = None

		self.sync = self.lib.__getattr__("sol_monocular_sync")
		self.sync.argtypes = None
		self.sync.restype = None

		self.opt_ = self.lib.__getattr__("sol_monocular_optimize")
		self.opt_.argtypes = [ctypes.c_int]
		self.opt_.restype = None

	def __init__(self, path="."):
		self.path = path
		self.create()

	def set_seed(self, s):
		arg = ctypes.c_uint64(s)
		self.seed_(arg)

	def optimize(self, level):
		arg = ctypes.c_int(level)
		self.opt_(arg)

	def set_IO(self, args):
		self.set_IO_.argtypes = [ndpointer(as_ctypes_type(x.dtype), flags="C_CONTIGUOUS") for x in args]
		self.set_IO_(*args)

	def run(self, args=None):
		if args:
			self.call.argtypes = [ndpointer(as_ctypes_type(x.dtype), flags="C_CONTIGUOUS") for x in args]
			self.call(*args)
		else:
			self.call_no_args()

	def close(self):
		dlclose_func = ctypes.CDLL(self.path + "/" + "libsol_monocular.so").dlclose
		dlclose_func.argtypes = (ctypes.c_void_p,)
		dlclose_func.restype = ctypes.c_int
		return dlclose_func(self.lib._handle)

# Initialize SOL model
depth_logger.info("Loading SOL optimized model...")
sol_model = sol_monocular(path=".")
sol_model.set_seed(271828) # optional
depth_logger.info("SOL model loaded successfully!")

def preprocess_frame(frame):
    resized_frame = cv2.resize(frame, (WIDTH, HEIGHT))
    normalized_frame = resized_frame.astype("float32") / 255.0
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

        # Preprocess the frame
        input_frame = preprocess_frame(frame)
        depth_logger.info(f"Preprocessed Frame: {input_frame.shape}, Min: {np.min(input_frame):.4f}, Max: {np.max(input_frame):.4f}")

        # Prepare input and output buffers for SOL model
        input_array = input_frame.astype(np.float32)  # Shape: (1, 256, 256, 3)
        output_array = np.zeros((1, 128, 128, 1), dtype=np.float32)  # Shape: (1, 128, 128, 1)
        vdims = np.ndarray((1), dtype=np.int64) # Shape: (1,), dtype: int64
        dp_args = [input_array, output_array, vdims] # Inputs, Outputs, VDims must be in this exact order!

        depth_logger.info(f"Input Array Shape: {input_array.shape}, Min: {np.min(input_array):.4f}, Max: {np.max(input_array):.4f}")
        depth_logger.info(f"Output Array Shape: {output_array.shape}, Min: {np.min(output_array):.4f}, Max: {np.max(output_array):.4f}")

        # Set input, output, and vdims arrays using set_IO
        sol_model.set_IO(dp_args)

        # Run SOL model
        sol_model.run() # (async)

        # Log output array after running the model
        depth_logger.info(f"Output Array After Run: Min: {np.min(output_array):.4f}, Max: {np.max(output_array):.4f}")

        # Extract depth map
        depth_map = output_array[0, :, :, 0]

        # Log if any NaN or Inf values are detected in the depth map
        if np.any(np.isnan(depth_map)) or np.any(np.isinf(depth_map)):
            depth_logger.warning(f"Depth map contains NaN or Inf values: {depth_map}")
            continue  # Skip this frame if the depth map is invalid

        # Log depth statistics
        min_depth = np.min(depth_map)
        max_depth = np.max(depth_map)
        avg_depth = np.mean(depth_map)
        std_depth = np.std(depth_map)
        median_depth = np.median(depth_map)

        frame_count += 1
        depth_logger.info(f"Frame {frame_count}:")
        depth_logger.info(f"  - Min Depth: {min_depth:.4f}")
        depth_logger.info(f"  - Max Depth: {max_depth:.4f}")
        depth_logger.info(f"  - Avg Depth: {avg_depth:.4f}")
        depth_logger.info(f"  - Std Depth: {std_depth:.4f}")
        depth_logger.info(f"  - Median Depth: {median_depth:.4f}")
        depth_logger.info("---------------------------------------------------")

        # Normalize depth map for visualization
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
    try:
        app.run(host='0.0.0.0', port=5000)
    finally:
        # Clean up SOL model
        sol_model.free()
        sol_model.close()