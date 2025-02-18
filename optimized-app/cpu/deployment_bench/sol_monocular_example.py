# Generated with SOL v0.7.0rc9
import numpy as np
import ctypes
from numpy.ctypeslib import ndpointer, as_ctypes_type
import time

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


def main():
	# Define Function Parameters (Inputs, Outputs, VDims)---------------------------------------------
	vdims = np.ndarray((1), dtype=np.int64)

	# Define input dimensions
	WIDTH, HEIGHT = 256, 256
	# WIDTH, HEIGHT = 512, 512
	
	# Create a random WIDTHxHEIGHT RGB image (1 image, 3 color channels), scaled to the float32 type.
	in__input_1 = np.random.rand(1, WIDTH, HEIGHT, 3, ).astype(np.float32)
	
	# Create a 256x256 image with one channel (e.g., depth map) and a batch size of 1, initialized to zeros in float32 format.
	out__0 = np.zeros((1, 256, 256, 1, ), dtype=np.float32)

	dp_args = [in__input_1, out__0, vdims] # Inputs, Outputs, VDims must be in this exact order!

	print("Input Data (first 5 values):", in__input_1.flatten()[:5])
	print("Before run, output buffer (first 5 values):", out__0.flatten()[:5])
	print("Input buffer shape:", in__input_1.shape)
	print("Output buffer shape:", out__0.shape)
	print("VDims shape:", vdims.shape)
	print("Input buffer address:", hex(in__input_1.__array_interface__['data'][0]))
	print("Output buffer address:", hex(out__0.__array_interface__['data'][0]))

	# Call Function and evaluate output---------------------------------------------
	lib = sol_monocular()
	lib.init() # optional
	lib.set_seed(271828) # optional

	####### Option 1: Run directly #######
	print("Running model...")
	lib.run(dp_args)
	print("Model run completed.")
	print("After run, output buffer (first 5 values):", out__0.flatten()[:5])
	print(f"Max_V: {np.max(out__0, axis=1)}\nMax_I: {np.argmax(out__0, axis=1)}")

	####### Option 2: Run directly #######
	# lib.set_IO(dp_args)
	# lib.optimize(level=2)
	# lib.run() # (async)
	# lib.get_output() # syncs
	# print(f"Max_V: {np.max(out__0, axis=1)}\nMax_I: {np.argmax(out__0, axis=1)}")

	# Free used data and close lib---------------------------------------------
	e = lib.close()
	lib.free()


if __name__ == "__main__":
	main()
