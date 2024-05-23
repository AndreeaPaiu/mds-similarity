import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial.distance import cdist

def best_fit_transform(A, B):
    '''
    Calculates the least-squares best-fit transform between corresponding 2D points A and B.
    '''
    assert len(A) == len(B)

    # translate points to their centroids
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    A -= centroid_A
    B -= centroid_B

    # rotation matrix
    H = np.dot(A.T, B)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)

    # special reflection case
    if np.linalg.det(R) < 0:
       Vt[1,:] *= -1
       R = np.dot(Vt.T, U.T)

    # translation
    t = centroid_B.T - np.dot(R,centroid_A.T)

    return R, t

def nearest_neighbor(src, dst):
    '''
    Find the nearest (Euclidean) neighbor in dst for each point in src
    '''
    print(src)
    print(dst)
    distances = cdist(src, dst, 'euclidean')
    indices = distances.argmin(axis=1)
    return dst[indices]

def icp(A, B, init_pose=None, max_iterations=20, tolerance=0.001):
    '''
    The Iterative Closest Point method
    '''
    src = np.array([A.T], copy=True).astype(float)
    dst = np.array([B.T], copy=True).astype(float)
    print(A.T)

    if init_pose is not None:
        src = np.dot(src, init_pose[0].T)
        src += init_pose[1]

    prev_error = 0

    for i in range(max_iterations):
        # find the nearest neighbors between the current source and destination points
        src = nearest_neighbor(src, dst)

        # compute the transformation between the current source and nearest destination points
        R, t = best_fit_transform(src, dst)

        # update the current source
        src = np.dot(R, src.T).T + t

        # check error
        mean_error = np.mean(np.linalg.norm(src - dst, axis=1))
        if np.abs(prev_error - mean_error) < tolerance:
            break
        prev_error = mean_error

    # calculate final transformation
    R, t = best_fit_transform(A, src)

    return R, t


# Generate two sets of 2D points as matrices
matrix1 = np.array([[1, 0], [2, 0], [2, 1], [1, 1], [1, 0]])
matrix2 = np.array([[1, 1], [2, 0], [3, 1], [2, 2], [1, 1]])
matrix3 = np.array([[1, 1], [3, 0], [3, 2], [1, 3], [1, 1]])
matrix4 = np.array([[1, 0], [3, 0], [3, 2], [1, 3], [1, 0]])

# # Perform Procrustes analysis to align matrix2 to matrix1
# R, s = orthogonal_procrustes(matrix1, matrix2)
#
# mtx2_aligned = matrix2@R.T

R, t = icp(matrix1, matrix2)

# Plot the original and aligned matrices
plt.figure()

# Plot original matrices
plt.subplot(1, 2, 1)
plt.plot(matrix1[:, 0], matrix1[:, 1], 'bo-', label='Matrix 1')
plt.plot(matrix2[:, 0], matrix2[:, 1], 'ro-', label='Matrix 2')
# plt.plot(matrix3[:, 0], matrix3[:, 1], 'go-', label='Matrix 3')
# plt.plot(matrix4[:, 0], matrix4[:, 1], 'yo-', label='Matrix 4')

plt.title('Original Matrices')
plt.legend()

mtx2 = np.dot(R, matrix2.T).T + t

# Plot aligned matrices
plt.subplot(1, 2, 2)
plt.plot(mtx2[:, 0], mtx2[:, 1], 'ro-', label='Matrix 2 Aligned')
# plt.plot(mtx1[:, 0], mtx1[:, 1], 'bo-', label='Matrix 1 Aligned')
# plt.plot(mtx2[:, 0], mtx2[:, 1], 'ro-', label='Matrix 2 Aligned')
# plt.plot(mtx3[:, 0], mtx3[:, 1], 'go-', label='Matrix 3 Aligned')
# plt.plot(mtx4[:, 0], mtx4[:, 1], 'yo-', label='Matrix 4 Aligned')

plt.title('Aligned Matrices')
plt.legend()

plt.show()