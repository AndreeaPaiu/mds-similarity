import numpy as np
import copy
from scipy.linalg import orthogonal_procrustes
from scipy.spatial import procrustes
import matplotlib.pyplot as plt
from numpy import linalg as LA

DATA_0_FILE = 'C:\\Users\\paiua\\Desktop\\work_project\\cercetare\\mds-similarity\\test-mds\\coords_mds_real_data\\data2-pixel-1.csv'
DATA_1_FILE = 'C:\\Users\\paiua\\Desktop\\work_project\\cercetare\\mds-similarity\\test-mds\\coords_mds_real_data\\data2-pixel-2.csv'
DATA_2_FILE = 'C:\\Users\\paiua\\Desktop\\work_project\\cercetare\\mds-similarity\\test-mds\\coords_mds_real_data\\data2-pixel-3.csv'

columns = ['idx', 'mds_x', 'mds_y', 'mds_z', 'x', 'y', 'z']
sim_data_0 = np.genfromtxt(DATA_0_FILE, delimiter=',', skip_header=1, names=columns, dtype=None)
sim_data_1 = np.genfromtxt(DATA_1_FILE, delimiter=',', skip_header=1, names=columns, dtype=None)
sim_data_2 = np.genfromtxt(DATA_2_FILE, delimiter=',', skip_header=1, names=columns, dtype=None)
Line1 = copy.deepcopy(sim_data_0)
Line2 = copy.deepcopy(sim_data_1)
Line3 = copy.deepcopy(sim_data_2)

matrix1 = []
realmatrix1 = []

for i in Line1:
    matrix1.append([i['mds_x'], i['mds_y']])
    realmatrix1.append([i['x'], i['y']])

matrix1 = np.array(matrix1)
realmatrix1 = np.array(realmatrix1)

media = np.mean(realmatrix1, axis = 0)
realmatrix1 = realmatrix1 - media


matrix2 = []
for i in Line2:
    matrix2.append([i['mds_x'], i['mds_y']])

matrix2 = np.array(matrix2)

matrix3 = []
for i in Line3:
    matrix3.append([i['mds_x'], i['mds_y']])

matrix3 = np.array(matrix3)

# Generate two sets of 2D points as matrices
# matrix1 = np.array([[1, 0], [2, 0], [2, 1], [1, 1], [1, 0]])
# matrix2 = np.array([[1, 1], [2, 0], [3, 1], [2, 2], [1, 1]])
# matrix3 = np.array([[1, 1], [3, 0], [3, 2], [1, 3], [1, 1]])
# matrix4 = np.array([[1, 0], [3, 0], [3, 2], [1, 3], [1, 0]])

# # Perform Procrustes analysis to align matrix2 to matrix1
mtx_r1 = np.array([realmatrix1[0], realmatrix1[12], realmatrix1[25], realmatrix1[37]])
mtx_r2 = np.array([matrix2[0], matrix2[12], matrix2[25], matrix2[37]])
R, s = orthogonal_procrustes(mtx_r1, mtx_r2)
print(s)
print(R.T)
mtx1 = realmatrix1
mtx2 = matrix2@R.T

# std1 = np.std(mtx_r2)
# std2 = np.std(mtx_r1)

std1 = np.std(mtx2)
std2 = np.std(mtx1)

# Scalăm prima matrice la aceeași scară cu a doua matrice
mtx2 = mtx2 / std1 * std2
print(std2 / std1)

# mtx1, mtx2, disparity = procrustes(matrix1, matrix2)
# print('1')
# print(mtx1)
# mtx1, mtx3, disparity = procrustes(matrix1, matrix3)
# print('2')
# print(mtx1)
# mtx1, mtx4, disparity = procrustes(matrix1, matrix4)
# print('3')
# print(mtx1)


# Plot the original and aligned matrices
fig = plt.figure()

# Plot original matrices
plt.subplot(1, 2, 1)
plt.plot(realmatrix1[:, 0], realmatrix1[:, 1], 'bo', label='Matrix 1')
plt.plot(matrix2[:, 0], matrix2[:, 1], 'ro', label='Matrix 2')
# plt.plot(matrix3[:, 0], matrix3[:, 1], 'go-', label='Matrix 3')
# plt.plot(matrix4[:, 0], matrix4[:, 1], 'yo-', label='Matrix 4')
plt.axis('equal')
plt.title('Original Matrices')
plt.legend()

# Plot aligned matrices
plt.subplot(1, 2, 2)
for i in range(len(mtx1)):
    plt.plot([mtx1[i][0], mtx2[i][0]], [mtx1[i][1], mtx2[i][1]], 'o-')
    plt.text(mtx1[i][0]-0.015, mtx1[i][1]+0.25, f"{i}")
    plt.text(mtx2[i][0]-0.050, mtx2[i][1]-0.25, f"{i}")
# plt.plot(mtx1[:, 0], mtx1[:, 1], 'bo', label='Matrix 1 Aligned')
# plt.plot(mtx2[:, 0], mtx2[:, 1], 'ro', label='Matrix 2 Aligned')
# plt.plot(mtx3[:, 0], mtx3[:, 1], 'go-', label='Matrix 3 Aligned')
# plt.plot(mtx4[:, 0], mtx4[:, 1], 'yo-', label='Matrix 4 Aligned')
plt.axis('equal')
plt.title('Aligned Matrices')
plt.legend()

plt.show()
fig.savefig("../test-mds/images/raport-2/procrustes_test_floor2_mds3.png", bbox_inches='tight')
