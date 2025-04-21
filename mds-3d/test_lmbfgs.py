import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from sklearn.metrics import euclidean_distances

# Known dissimilarities (symmetric matrix)
data = np.array([[8, 5], [0, 10], [-8, 5], [-8, -5], [0, -10], [8, -5], [1, 1]]) # hexagon
D = euclidean_distances(data)
# D = np.array([
#     [0., 9.43398113, 16., 18.86796226, 17., 10.],
#     [9.43398113, 0., 9.43398113, 17., 20., 17.],
#     [16., 9.43398113, 0., 10., 17., 18.86796226],
#     [18.86796226, 17., 10., 0., 9.43398113, 16.],
#     [17., 20., 17., 9.43398113, 0., 9.43398113],
#     [10., 17., 18.86796226, 16., 9.43398113, 0.]
# ])

# Known positions for the first 3 points
known = np.array([
    [8, 5],
    [0, 10]
])

# Number of unknown points
n_unknown = 5

# Function to compute the stress (difference between distances and dissimilarities)
def stress(unknown_flat):
    unknown = unknown_flat.reshape((n_unknown, 2))
    all_points = np.vstack([known, unknown])
    dist_matrix = cdist(all_points, all_points)
    return np.sum((dist_matrix - D)**2)

# Initial guess for unknown points (random or roughly spread)
init_guess = np.random.rand(n_unknown * 2) * 20 - 10

# Run optimization
result = minimize(stress, init_guess, method='L-BFGS-B')

# Extract positions
estimated = result.x.reshape((n_unknown, 2))

# Combine known and estimated positions
all_points = np.vstack([known, estimated])

# Plotting the points
plt.figure(figsize=(8, 6))
plt.scatter(all_points[:, 0], all_points[:, 1], color='blue', label='Points')
plt.scatter(known[:, 0], known[:, 1], color='red', label='Known Points', zorder=5)
for i, (x, y) in enumerate(known):
    plt.text(x, y, f'P{i+1}', fontsize=12, ha='right', color='red')
for i, (x, y) in enumerate(estimated):
    plt.text(x, y, f'P{i+4}', fontsize=12, ha='left', color='blue')
plt.title("Reconstructed Points from Dissimilarities")
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.legend()
plt.grid(True)
plt.show()
