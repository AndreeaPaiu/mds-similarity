import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial.distance import canberra, chebyshev, cityblock, correlation, euclidean, jensenshannon, mahalanobis, \
    minkowski, seuclidean, sqeuclidean, dice, hamming, jaccard, kulczynski1, rogerstanimoto, russellrao, sokalmichener, \
    sokalsneath, yule, pdist, cdist, squareform, directed_hausdorff
from sklearn.manifold import MDS
import matplotlib.patches as mpatches
from compare_locations import compare_locations

def plot_mds_braycurtis(collections, aps):
    # Matricea de similarități între produse
    similarities = np.empty((len(collections), len(collections)))

    for i in range(len(collections)):
        for j in range(len(collections)):
            if 'wifi' not in collections[i] or 'wifi' not in collections[j]:
                similarities[i][j] = 1
                continue

            similarities[i][j] = compare_locations(collections[i], collections[j], all_aps=aps)

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

    plt.scatter(coordinates[:, 0], coordinates[:, 1])
    for i, (x, y) in enumerate(coordinates):
        plt.scatter(x, y, color=colors[int(collections[i]['floor'])], label=f'Line {i+1}')
        plt.text(x + .03, y + .03, i, fontsize=9)

    plt.xlabel('Dimensiune 1')
    plt.ylabel('Dimensiune 2')
    plt.title('[Braycurtis][all aps] Reprezentarea a unui etaj folosind MDS')
    handles = []
    for i in range(0, int(collections[len(collections) - 1]['floor']) + 1):
        handles.append(mpatches.Patch(color=colors[i], label='etaj' + str(i)))

    plt.legend(handles=handles)
    plt.grid()
    plt.show()
    # fig.savefig(f"images/label_mds_braycurtis.svg", bbox_inches='tight')
    # fig.savefig(f"images/mds_braycurtis_all_floors.svg", bbox_inches='tight')
