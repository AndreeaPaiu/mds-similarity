import random

import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial.distance import braycurtis, cosine, correlation, yule
from scipy.stats._mstats_basic import pearsonr
from sklearn.cluster import KMeans
from sklearn.manifold import MDS
import matplotlib.patches as mpatches

from compare_locations import compare_locations
from compute_mds_wifi_similarity import *
from compute_mds_cartesian import *
from pearsonr_similarity import *
from helpers import *


# compute label floor
# compute 2D each floor
# ajust floors with center in 0
# display 3D

colors = []
for color in matplotlib.colors.TABLEAU_COLORS:
    colors.append(color)

def plot_mds_z(collections, simil_method=cosine, n_dim=2, xlabel='Dimensiunea1', ylabel='Dimensiunea2',
             zlabel='Dimnesiunea3', title='', file_name='images/plot.svg', selection='All', add_label=False, check_one=False, plot_slope=False, print_angle=False, type_data='wifi', n_clusters=2):
    # pentru a fi sigura ca punctele pot fi random
    random.shuffle(collections)
    # calculez etajul
    collections = add_floor_label_to_collection(
        collections,
        simil_method=cosine,
        n_clusters=n_clusters,
        selection=selection,
        type_data=type_data,
        n_dim=n_dim
    )

    # separ pe etaje
    floors_collections = {}
    for collection in collections:
        if collection['floor'] not in floors_collections:
            floors_collections[collection['floor']] = []
        floors_collections[collection['floor']].append(collection)

    # asamblez atajele
    coordinates = []
    for i in floors_collections:
        coordinates = coordinates + compute_2D_mds(
            floors_collections[i],
            simil_method=simil_method,
            selection=selection,
            type_data=type_data
        )

    # Vizualizarea rezultatelor
    fig = plt.figure()
    plt.axis('equal')
    # # plt.axis([-1, 1, -1, 1])
    # # ax = fig.gca()
    # # ax.set_autoscale_on(False)
    if n_dim == 3:
        ax = fig.add_subplot(111, projection='3d')

    # if n_dim == 2:
    #     plt.xlabel(xlabel)
    #     plt.ylabel(ylabel)
    #     for i, (x, y) in enumerate(coordinates):
    #         plt.scatter(x, y, color=colors[int(collections[i]['floor'])], label=f'Line {i + 1}')
    #
    #     if add_label:
    #         for i, (x, y) in enumerate(coordinates):
    #             plt.text(x + .03, y + .03, collections[i]['floor'], fontsize=9)
    #
    if n_dim == 3:
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_zlabel(zlabel)

        for i, (x, y, z, f) in enumerate(coordinates):
            ax.scatter(x, y, z, color=colors[z], label=f'Line {z}')

        if add_label:
            for i, (x, y, z, f) in enumerate(coordinates):
                # ax.text(x + .03, y + .03,  z + .03, f" {i % len(floors_collections[collections[i]['floor']])}", fontsize=9)
                ax.text(x + .03, y + .03, z + .03, f" {f}",
                        fontsize=9)

    plt.title(title)
    handles = []
    for i in range(0, n_clusters):
        handles.append(mpatches.Patch(color=colors[i], label=f'etaj{i}'))

    plt.legend(handles=handles)
    plt.grid()
    plt.show()
    fig.savefig(file_name, bbox_inches='tight')

