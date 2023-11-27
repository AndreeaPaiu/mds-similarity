import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from sklearn.manifold import MDS
import matplotlib.patches as mpatches
from compare_locations import compare_locations


def plot_mds_braycurtis(collections):
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

    colors = []
    for color in matplotlib.colors.TABLEAU_COLORS:
        colors.append(color)

    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']

    plt.scatter(coordinates[:, 0], coordinates[:, 1])
    for i, (x, y) in enumerate(coordinates):
        plt.plot(x, y, colors[int(collections[i]['floor'])] + 'o')

    plt.xlabel('Dimensiune 1')
    plt.ylabel('Dimensiune 2')
    plt.title('Reprezentarea similarității între produse folosind MDS')
    handles = []
    for i in range(0, int(collections[len(collections) - 1]['floor']) + 1):
        handles.append(mpatches.Patch(color=colors[i], label='etaj' + str(i)))

    plt.legend(handles=handles)
    plt.grid()
    plt.show()
    fig.savefig(f"images/mds_braycurtis.svg", bbox_inches='tight')
    # fig.savefig(f"images/mds_braycurtis_all_floors.svg", bbox_inches='tight')
