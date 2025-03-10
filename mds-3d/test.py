from sklearn.manifold import MDS
import numpy as np
import matplotlib.pyplot as plt
from _mdsap import *
from sklearn.metrics import euclidean_distances
import matplotlib.patches as mpatches
import pandas as pd


def center_matrix(points, init_points, axis=0):
    points = points.astype(float)
    init_points = init_points.astype(float)
    meanPoint = np.mean(points, axis=0)
    points -= meanPoint
    meanPoint = np.mean(init_points, axis=0)
    init_points -= meanPoint

    return points, init_points

def get_limits(points, addition=0):
    df = pd.DataFrame(points)

    return [[df.min()[0] - addition, df.max()[0] + addition], [df.min()[1] - addition, df.max()[1] + addition]]
# Example data
# data = [[0.0, 0.0], [0.1, 0.1], [0.2, 0.3], [0.4,0.7]]
# data = np.array([[0, 0], [3, 3], [2, 2], [1, 1]]) # linie
# data = np.array([[0,0], [0, 2], [2,2], [2,0]]) # patrat
data = np.array([[8, 5], [0, 10], [-8, 5], [-8, -5], [0, -10], [8, -5]]) # hexagon
# data = np.array([[8, 5], [0, 10], [2, 3], [1, -4], [-2, -10], [-7, -4]]) # random
# data = np.array([[0,0], [0, 2], [2,2], [2,0], [-1,-1],[-1,3], [3,3], [3, -1]]) # 2 patrat

dist_data = euclidean_distances(data)
# Custom initial configuration
# init_config = np.array([[0.0, 0.0], [0.1, 0.1]])
# init_config = np.array([[0, 0], [3, 3]]) #linie
# init_config = np.array([[0, 0], [0, 2]]) # patrat
init_config = np.array([[8, 5], [0, 10], [-8, 5]]) # hexagon
# init_config = np.array([[8, 5], [0, 10]]) # random
# init_config = np.array([[0, 0], [0, 2], [2,2]]) # 2 patrat
# init_config = None


data, init_config = center_matrix(data, init_config, 0)
limit_points = get_limits(init_config)
dist_init_config = euclidean_distances(init_config)

# s = 0
# iter = 0
# for i in range(len(dist_init_config)):
#     for j in range(len(dist_init_config)):
#         print(i, j)
#         if dist_init_config[i][j] != 0:
#             s += dist_init_config[i][j]/dist_data[i][j]
#             iter +=1
#
# s = s / iter
# print(f's = {s}')
# data = np.multiply(data, s)
# Create MDS instance with init parameter
mds = MDSAP(n_components=2, random_state=0, max_iter=300, eps=1e-6, random_limit_points=limit_points)
# w=np.array([[1, 1, 1, 1],[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]])
# Fit the model
transformed_data = mds.fit_transform(data, init=init_config)
# init_config = np.array([[0, 0], [0, 2]]) # 2 patrat
# _, init_config = center_matrix(transformed_data, init_config, 0)
# print(init_config)
# exit()
# transformed_data = mds.fit_transform(transformed_data, init=init_config)


print(transformed_data)
fig = plt.figure()
plt.axis('equal')
_, ax = plt.subplots()

for i in range(len(data)):
    if i < len(init_config):
        if init_config[i][0] == data[i][0] and init_config[i][1] == data[i][1]:
            # plt.scatter(data[i][0], data[i][1], fillstyle='left', color='green', markerfacecoloralt='red')
            ax.plot(data[i][0], data[i][1], 'o',  # middle part
                    fillstyle='left', color='green', markerfacecoloralt='red')
            plt.text(data[i][0] + .03, data[i][1] + .03, i, fontsize=9)
        else:
            plt.scatter(init_config[i][0], init_config[i][1], color='red')
            plt.text(init_config[i][0] + .03, init_config[i][1] + .03, i, fontsize=9)
            plt.scatter(data[i][0], data[i][1], color='green')
            plt.text(data[i][0] + .03, data[i][1] + .03, i, fontsize=9)
    else:
        plt.scatter(data[i][0], data[i][1], color='green')
        plt.text(data[i][0] + .03, data[i][1] + .03, i, fontsize=9)

# exit()
# for i, (x, y) in enumerate(init_config):
#     plt.scatter(x, y, color='red')
#     plt.text(x + .03, y + .03, i, fontsize=9)
#
# for i, (x, y) in enumerate(data):
#     plt.scatter(x, y, color='green')
#     plt.text(x + .03, y + .03, i, fontsize=9)

for i, (x, y) in enumerate(transformed_data):
    plt.scatter(x, y, color='blue')
    plt.text(x + .03, y + .03, i, fontsize=9)

handles = []
handles.append(mpatches.Patch(color='red', label=f'init data'))
handles.append(mpatches.Patch(color='green', label=f'our data'))
handles.append(mpatches.Patch(color='blue', label=f'transformed data'))

plt.legend(handles=handles)
ax.set_aspect('equal', adjustable='box')

print(f'init config distances: {euclidean_distances(init_config)}')
print(f'transformed data distances: {euclidean_distances(transformed_data)}')
print(f'init config distances: {euclidean_distances(data)}')
plt.grid()
plt.show()
# fig.savefig(f'test_mds_8_points_in_2_square_keep_init.png', bbox_inches='tight')
# fig.savefig(f'test_mds_8_points_in_2_square.png', bbox_inches='tight')