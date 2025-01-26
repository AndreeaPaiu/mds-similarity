from sklearn.manifold import MDS
import numpy as np
import matplotlib.pyplot as plt
from _mdsap import *
from sklearn.metrics import euclidean_distances


# Example data
data = [[0.0, 0.0], [0.1, 0.1], [0.2, 0.3], [0.4,0.7]]
# data = [[0, 0], [1, 1], [2, 2], [3, 3]] # linie
# data = [[0,0], [0,2], [2,2], [2,0]] # patrat

dist_data = euclidean_distances(data)
# Custom initial configuration
# init_config = np.array([[0.0, 0.0], [0.1, 0.1]])
init_config = np.array([[0, 0], [1, 1]]) #linie
# init_config = np.array([[0, 0], [0, 2]]) # patrat

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
mds = MDSAP(n_components=2, random_state=0)

# Fit the model
transformed_data = mds.fit_transform(data, init=init_config)
print(transformed_data)
fig = plt.figure()
plt.axis('equal')
for i, (x, y) in enumerate(data):
    plt.scatter(x, y, color='red')

for i, (x, y) in enumerate(transformed_data):
    plt.scatter(x, y, color='blue')
plt.grid()
plt.show()
