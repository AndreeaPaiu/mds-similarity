import random

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Generare aleatorie a unui set de puncte
np.random.seed(0)
X = np.random.rand(100,2)
random.shuffle(X)

# Antrenarea algoritmului <link>K-means</link> pentru a clasifica punctele în 2 grupuri
kmeans = KMeans(n_clusters=2, random_state=0).fit(X)

# Afișarea rezultatelor clasificării
plt.scatter(X[:, 0], X[:, 1], c=kmeans.labels_, cmap='viridis')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], marker='x', s=200, c='r')
plt.show()