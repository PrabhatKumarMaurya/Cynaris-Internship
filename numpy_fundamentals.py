import numpy as np

# 1. 1D Array
a = np.array([10, 20, 30, 40])

# 2. 2D Array
b = np.array([[1, 2], [3, 4]])

# 3. 3D Array
c = np.array([[[1, 2], [3, 4]]])

print("1D Shape:", a.shape)
print("2D Shape:", b.shape)
print("3D Shape:", c.shape)

# 4. Vectorized Operation
print("Addition:", a + 10)

# 5. Broadcasting
print("Broadcasting:", a * 2)

# 6. Matrix Multiplication
print("Matrix Multiplication:\n", b @ b)

# 7. Statistics
print("Mean:", np.mean(a))
print("Standard Deviation:", np.std(a))
print("Correlation:", np.corrcoef(a, [20, 30, 40, 50]))