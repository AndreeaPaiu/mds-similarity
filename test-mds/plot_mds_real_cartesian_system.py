import math

import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from sklearn.manifold import MDS
import matplotlib.patches as mpatches



def plot_mds_real_cartesian_system(collections):
    # Matricea de similarități între produse
    similarities = np.empty((len(collections), len(collections)))

    for i in range(len(collections)):
        for j in range(len(collections)):
            if 'real_coordinates' not in collections[i] or 'real_coordinates' not in collections[j]:
                similarities[i][j] = 100
                continue

            # a floor
            similarities[i][j] = math.dist(collections[i]['real_coordinates'][0:2], collections[j]['real_coordinates'][0:2])

            # all floors
            # similarities[i][j] = math.dist(collections[i]['real_coordinates'][0:3], collections[j]['real_coordinates'][0:3])

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

    # plt.scatter(coordinates[:, 0], coordinates[:, 1])
    for i, (x, y) in enumerate(coordinates):
        plt.plot(x, y, colors[int(collections[i]['floor_id'])] + 'o')

    for i, (x, y) in enumerate(coordinates):
        plt.text(x + .03, y + .03, collections[i]['label_id'], fontsize=9)

    plt.xlabel('Dimensiune 1')
    plt.ylabel('Dimensiune 2')
    plt.title('Reprezentarea similarității între produse folosind MDS')
    handles = []
    for i in range(0, int(collections[len(collections) - 1]['floor_id']) + 1):
        handles.append(mpatches.Patch(color=colors[i], label='etaj' + str(i)))

    plt.legend(handles=handles)
    plt.grid()
    plt.show()
    fig.savefig(f"images/mds-cartesian_system.svg", bbox_inches='tight')
    # fig.savefig(f"images/mds-cartesian_system-all-floor.svg", bbox_inches='tight')

