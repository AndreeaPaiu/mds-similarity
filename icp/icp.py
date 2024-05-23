#%%
import copy

import numpy as np
import matplotlib.pyplot as plt

DATA_0_FILE = 'C:\\Users\\paiua\\Desktop\\work_project\\cercetare\\mds-similarity\\test-mds\\coords_mds_real_data\\data0.csv'
DATA_1_FILE = 'C:\\Users\\paiua\\Desktop\\work_project\\cercetare\\mds-similarity\\test-mds\\coords_mds_real_data\\data1.csv'

columns = ['idx', 'mds_x', 'mds_y', 'mds_z', 'x', 'y', 'z']
sim_data_0 = np.genfromtxt(DATA_0_FILE, delimiter=',', skip_header=1, names=columns, dtype=None)
sim_data_1 = np.genfromtxt(DATA_1_FILE, delimiter=',', skip_header=1, names=columns, dtype=None)
#%%
# icp_known_corresp: performs icp given that the input datasets
# are aligned so that Line1(:, QInd(k)) corresponds to Line2(:, PInd(k))
def icp_known_corresp(Line1, Line2, QInd, PInd):
    Q = Line1[:, QInd]
    P = Line2[:, PInd]

    MuQ = compute_mean(Q)
    MuP = compute_mean(P)

    W = compute_W(Q, P.copy(), MuQ, MuP)

    [R, t] = compute_R_t(W, MuQ, MuP)

    # Compute the new positions of the points after
    # applying found rotation and translation to them
    NewLine = R @ P + t[:, None]

    E = compute_error(Q, NewLine)
    return [NewLine, E, R, t]

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
def show_figure(Line1, Line2, title=""):
    plt.figure()
    # plt.scatter(Line1[0], Line1[1], marker='o', s=2, label='Line 1')
    plt.scatter(Line2[0], Line2[1], marker='x', s=4, label='Line 2')
    plt.title(title)

    # plt.xlim([-1, 1])
    # plt.ylim([-1, 1])
    plt.axis('equal')
    plt.legend()  

    plt.show()


# initialize figure
def init_figure():
    fig = plt.gcf()
    fig.show()
    fig.canvas.draw()

    line1_fig = plt.scatter([], [], marker='o', s=2, label='Line 1')
    line2_fig = plt.scatter([], [], marker='o', s=1, label='Line 2')
    # plt.title(title)
    plt.xlim([-8, 8])
    plt.ylim([-8, 8])
    plt.legend()

    return fig, line1_fig, line2_fig


# update_figure: show the current state of the lines
def update_figure(fig, line1_fig, line2_fig, Line1, Line2, hold=False):
    line1_fig.set_offsets(Line1.T)
    line2_fig.set_offsets(Line2.T)
    if hold:
        plt.show()
    else:
        fig.canvas.flush_events()
        fig.canvas.draw()
        plt.pause(0.5)


Data = np.load('icp_data.npz')
Line1 = Data['LineGroundTruth']
Line2 = Data['LineMovedCorresp']
# print(type(Line1))
# Line1 = copy.deepcopy(sim_data_0)
# Line2 = copy.deepcopy(sim_data_1)
#
# x_1 = []
# y_1 = []
# for i in Line1:
#     x_1 = x_1 + [i['mds_x']]
#     y_1 = y_1 + [i['mds_y']]
#
# Line1 = np.array([x_1, y_1])
#
# x_2 = []
# y_2 = []
# for i in Line2:
#     x_2 = x_2 + [i['mds_x']]
#     y_2 = y_2 + [i['mds_y']]

# Line2 = np.array([x_2, y_2])

# Show the initial positions of the lines
show_figure(Line1, Line2, "initial")


# We assume that the there are 1 to 1 correspondences for this data
l = len(Line1[0])
QInd = np.arange(1, l, int(l/2))
PInd = np.arange(1, l, int(l/2))
# QInd = [1, 2, 3, 4, 5, 6]
# PInd = [1, 2, 3, 4, 5, 6]



# Perform icp given the correspondences
[Line3, E, R, t] = icp_known_corresp(Line1, Line2, QInd, PInd)
print(Line3)

# Show the adjusted positions of the lines
show_figure(Line1, Line3, "registered")

# print the error
print('Error value is: ', E, "\nRotation:\n", R, "\nTranslation:\n", t)


# %%
