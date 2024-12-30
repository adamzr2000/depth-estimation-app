import tensorflow as tf
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

# Perform computation on GPU or fallback to CPU
try:
    if gpus:
        with tf.device('/GPU:0'):
            print("Performing a small computation on the GPU...")
            result = tf.reduce_sum(tf.random.normal([1000, 1000]))
            print(f"GPU Computation Result: {result.numpy()}")
    else:
        print("Performing a computation on the CPU...")
        with tf.device('/CPU:0'):
            result = tf.reduce_sum(tf.random.normal([1000, 1000]))
            print(f"CPU Computation Result: {result.numpy()}")
except Exception as e:
    print(f"Error during computation: {e}")

print("=== Test Complete ===")
