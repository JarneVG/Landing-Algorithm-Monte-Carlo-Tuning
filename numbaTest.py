import numpy as np
import time
from numba import jit

# Generate large input array
x = np.linspace(-10, 10, 10_000_000)

# --- 1. Pure Python version
def slow_function(x):
    result = []
    for val in x:
        result.append(np.sin(val) * np.exp(-val**2))
    return result

# --- 2. NumPy vectorized version
def numpy_function(x):
    return np.sin(x) * np.exp(-x**2)

# --- 3. Numba JIT version (on a loop-based function)
@jit(nopython=True)
def numba_function(x):
    result = np.empty_like(x)
    for i in range(x.shape[0]):
        result[i] = np.sin(x[i]) * np.exp(-x[i]**2)
    return result

# --- Timing helper
def time_function(func, x, label):
    start = time.time()
    y = func(x)
    end = time.time()
    print(f"{label:20s}: {end - start:.4f} seconds")
    return y

# --- Run and time each version
print("Timing results:")
y1 = time_function(slow_function, x, "Pure Python")
y2 = time_function(numpy_function, x, "NumPy")
y3 = time_function(numba_function, x, "Numba JIT")
