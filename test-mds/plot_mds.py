import numpy as np
from matplotlib import pyplot as plt
from sklearn.manifold import MDS

from compare_locations import compare_locations


def plot_mds(collections):

    # Matricea de similarități între produse
    similarities = np.empty((len(collections), len(collections)))

    for i in range(len(collections)):
        for j in range(len(collections)):
            if 'wifi' not in collections[i] or 'wifi' not in collections[j]:
                similarities[i][j] = 1
                continue

            similarities[i][j] = compare_locations(collections[i], collections[j])

    # Crearea unui obiect MDS cu 2 dimensiuni
    mds = MDS(n_components=2, dissimilarity='precomputed')

    # Aplicarea MDS pe matricea de similarități
    coordinates = mds.fit_transform(similarities)

    # Vizualizarea rezultatelor
    fig = plt.figure()
    plt.axis('equal')
    plt.scatter(coordinates[:, 0], coordinates[:, 1])
    for i, (x, y) in enumerate(coordinates):
        plt.plot(x, y)
    plt.xlabel('Dimensiune 1')
    plt.ylabel('Dimensiune 2')
    plt.title('Reprezentarea similarității între produse folosind MDS')
    plt.grid()
    plt.show()
    fig.savefig(f"plot_mds.pdf", bbox_inches='tight')
