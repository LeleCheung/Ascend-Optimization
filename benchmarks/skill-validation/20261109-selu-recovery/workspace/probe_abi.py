import sys, inspect
import triton
print("python", sys.version.split()[0])
print("triton", triton.__version__, triton.__file__)
from triton.backends.ascend.driver import NPULauncher
print("launch sig:", inspect.signature(NPULauncher.launch))
print("launch doc:", NPULauncher.launch.__doc__)
try:
    print(inspect.getsource(NPULauncher.launch))
except Exception as e:
    print("src err", e)
