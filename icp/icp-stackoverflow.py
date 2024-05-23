import copy
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import procrustes
from scipy.linalg import orthogonal_procrustes


DATA_0_FILE = 'C:\\Users\\paiua\\Desktop\\work_project\\cercetare\\mds-similarity\\test-mds\\coords_mds_real_data\\data2-pixel-1.csv'
DATA_1_FILE = 'C:\\Users\\paiua\\Desktop\\work_project\\cercetare\\mds-similarity\\test-mds\\coords_mds_real_data\\data2-pixel-2.csv'
DATA_2_FILE = 'C:\\Users\\paiua\\Desktop\\work_project\\cercetare\\mds-similarity\\test-mds\\coords_mds_real_data\\data2-pixel-3.csv'

columns = ['idx', 'mds_x', 'mds_y', 'mds_z', 'x', 'y', 'z']
sim_data_0 = np.genfromtxt(DATA_0_FILE, delimiter=',', skip_header=1, names=columns, dtype=None)
sim_data_1 = np.genfromtxt(DATA_1_FILE, delimiter=',', skip_header=1, names=columns, dtype=None)
sim_data_2 = np.genfromtxt(DATA_2_FILE, delimiter=',', skip_header=1, names=columns, dtype=None)

def center_data(data):
    # am centrat datele reale
    #create function to center data
    center_function = lambda x: x - x.mean()

    #apply function to original NumPy array
    data[0] = center_function(data[0])
    data[1] = center_function(data[1])
    return data

# icp_known_corresp: performs icp given that the input datasets
# are aligned so that Line1(:, QInd(k)) corresponds to Line2(:, PInd(k))
def icp_known_corresp(Line1, Line2, QInd, PInd):
    Q = Line1[:, QInd]
    P = Line2[:, PInd]

    MuQ = compute_mean(Q)
    MuP = compute_mean(P)

    W = compute_W(Q, P, MuQ, MuP)

    [R, t] = compute_R_t(W, MuQ, MuP)

    # Compute the new positions of the points after
    # applying found rotation and translation to them
    NewLine = R @ P

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # If i don't add t to the NewLine the results are good. #
    # If i add t, there will be a gap between two curves.   #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

#     NewLine[0, :] += t[0]
#     NewLine[1, :] += t[1]

    E = compute_error(Q, NewLine)
    return [NewLine, E]

# compute_W: compute matrix W to use in SVD
def compute_W(Q, P, MuQ, MuP):
    Q[0, :] -= MuQ[0]
    Q[1, :] -= MuQ[1]
    P[0, :] -= MuP[0]
    P[1, :] -= MuP[1]
    return Q @ P.T


# compute_R_t: compute rotation matrix and translation vector
# based on the SVD as presented in the lecture
def compute_R_t(W, MuQ, MuP):
    U,S,V = np.linalg.svd(W)
    R = U @ V
    t = MuQ - R @ MuP
    return [R, t]

# compute_mean: compute mean value for a [M x N] matrix
def compute_mean(M):
    return np.mean(M, axis = 1)

# compute_error: compute the icp error
def compute_error(Q, OptimizedPoints):
    E = Q - OptimizedPoints
    return np.sqrt(np.sum(E**2))

# simply show the two lines
def show_figure(Line1, Line2, Line3):

    for i in range(len(Line1[0])):
        plt.plot([Line1[0][i], Line2[0][i]], [Line1[1][i], Line2[1][i]], 'o-')
        plt.text(Line1[0][i]-0.015, Line1[1][i]+0.25, f"{i}")
        plt.text(Line2[0][i]-0.050, Line2[1][i]-0.25, f"{i}")

#     plt.scatter(Line1[0], Line1[1], marker='o', s=3, label='Set of Points 1')
#     plt.scatter(Line2[0], Line2[1], marker='o', s=2, label='Set of Points 2')
#     plt.scatter(Line3[0], Line3[1], marker='o', s=1, label='Set of Points 3')

#     plt.xlim([-8, 8])
#     plt.ylim([-8, 8])
    plt.axis('equal')
    plt.legend()

Data = np.load('icp_data.npz')
# Line1 = Data['LineGroundTruth']
# Line2 = Data['LineMovedCorresp']
Line1 = copy.deepcopy(sim_data_0)
Line2 = copy.deepcopy(sim_data_1)
Line3 = copy.deepcopy(sim_data_2)


x_1 = []
y_1 = []
rx_1 = []
ry_1 = []
for i in Line1:
    x_1 = x_1 + [i['mds_x']]
    y_1 = y_1 + [i['mds_y']]
    rx_1 = rx_1 + [i['x']]
    ry_1 = ry_1 + [i['y']]

Line1 = np.array([x_1, y_1])
RealLine1 = np.array([rx_1, ry_1])

center_data(RealLine1)


x_2 = []
y_2 = []
for i in Line2:
    x_2 = x_2 + [i['mds_x']]
    y_2 = y_2 + [i['mds_y']]

Line2 = np.array([x_2, y_2])

x_3 = []
y_3 = []
for i in Line3:
    x_3 = x_3 + [i['mds_x']]
    y_3 = y_3 + [i['mds_y']]

Line3 = np.array([x_3, y_3])

# We assume that the there are 1 to 1 correspondences for this data
QInd = np.arange(len(RealLine1[0]))
PInd = np.arange(len(Line1[0]))

# Perform icp given the correspondences
[Line2_r, E] = icp_known_corresp(RealLine1, Line1, QInd, PInd)

QInd = np.arange(len(RealLine1[0]))
PInd = np.arange(len(Line3[0]))

# Perform icp given the correspondences
[Line3_r, E] = icp_known_corresp(RealLine1, Line3, QInd, PInd)

fig = plt.figure()
# Show the initial positions of the lines

plt.subplot(1, 2, 1)

# show_figure(Line1, RealLine1, Line3)
plt.scatter(RealLine1[0], RealLine1[1], marker='o', s=3, label='Set of Points 1')
plt.scatter(Line1[0], Line1[1], marker='o', s=2, label='Real Set of Points 2')
plt.axis('equal')
plt.title("initial positions of the points")
plt.legend()
# Show the adjusted positions of the lines

plt.subplot(1, 2, 2)
show_figure(RealLine1, Line2_r, Line3_r)

plt.title("adjusted positions of the points")
plt.legend()

plt.show()
fig.savefig("../test-mds/images/raport-2/icp_test_floor2_mds1_real_coord.png", bbox_inches='tight')

# print the error
print('Error value is: ', E)


