import numpy as np
from scipy.linalg import orthogonal_procrustes
from scipy.spatial import procrustes
import matplotlib.pyplot as plt
from numpy import linalg as LA

# Generate two sets of 2D points as matrices
matrix1 = np.array([[1, 0], [2, 0], [2, 1], [1, 1], [1, 0]])
matrix2 = np.array([[1, 1], [2, 0], [3, 1], [2, 2], [1, 1]])
matrix3 = np.array([[1, 0], [3, 0], [3, 2], [1, 2], [1, 0]])

# # Perform Procrustes analysis to align matrix2 to matrix1
# R, s = orthogonal_procrustes(matrix1, matrix2)
#
# mtx2_aligned = matrix2@R.T

mtx1, mtx2, disparity = procrustes(matrix1, matrix2)
mtx1, mtx3, disparity = procrustes(matrix1, matrix3)

# Plot the original and aligned matrices
plt.figure()

# Plot original matrices
plt.subplot(1, 2, 1)
plt.plot(matrix1[:, 0], matrix1[:, 1], 'bo-', label='Matrix 1')
plt.plot(matrix2[:, 0], matrix2[:, 1], 'ro-', label='Matrix 2')
plt.plot(matrix3[:, 0], matrix3[:, 1], 'go-', label='Matrix 3')
plt.title('Original Matrices')
plt.legend()

# Plot aligned matrices
plt.subplot(1, 2, 2)
plt.plot(mtx1[:, 0], mtx1[:, 1], 'bo-', label='Matrix 1 Aligned')
plt.plot(mtx2[:, 0], mtx2[:, 1], 'ro-', label='Matrix 2 Aligned')
plt.plot(mtx3[:, 0], mtx3[:, 1], 'go-', label='Matrix 3 Aligned')
plt.title('Aligned Matrices')
plt.legend()

plt.show()