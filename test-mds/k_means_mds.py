import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial.distance import squareform, pdist
from sklearn.cluster import KMeans
from sklearn.manifold import MDS

from compare_locations import *


def k_means_mds(collections):
    similarities = np.empty((len(collections), len(collections)))

    for i in range(len(collections)):
        for j in range(len(collections)):
            if 'wifi' not in collections[i] or 'wifi' not in collections[j]:
                similarities[i][j] = 1
                continue

            similarities[i][j] = compare_locations(collections[i], collections[j])

    # Numărul de grupuri dorite
    k = 8
    # Aplicarea analizei MDS pentru reducerea dimensionalității
    mds = MDS(n_components=2, dissimilarity='precomputed')
    puncte_mds = mds.fit_transform(similarities)

    # Inițializarea și aplicarea algoritmului <link>k-means</link> pe punctele MDS
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(puncte_mds)

    # Atribuirea punctelor MDS la grupuri
    grupuri = kmeans.labels_

    # Obținerea centroizilor grupurilor în coordonate MDS
    centroizi_mds = kmeans.cluster_centers_

    centroizi_invers = mds.inverse_transform(kmeans.cluster_centers_)


    # Afișarea rezultatelor
    plt.scatter(puncte_mds[:, 0], puncte_mds[:, 1], c=grupuri)
    plt.scatter(centroizi_mds[:, 0], centroizi_mds[:, 1], c='red', marker='X')
    plt.title("Clasificarea punctelor în grupuri utilizând <link>k-means</link> și MDS")
    plt.xlabel("Componenta 1")
    plt.ylabel("Componenta 2")
    plt.show()