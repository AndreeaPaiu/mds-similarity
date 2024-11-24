from sklearn.manifold import MDS
from sklearn.decomposition import PCA
from datetime import datetime
from scipy.spatial.distance import cosine
from compare_locations import compare_locations

import math
import numpy as np
from matplotlib import pyplot as plt
import matplotlib

colors = []
for color in matplotlib.colors.TABLEAU_COLORS:
    colors.append(color)

def get_collection_components(collections = [], type='wifi'):
    if type == 'wifi':
        return get_wifi_rssi(collections)
    elif type == 'cartesian':
        return get_cartesian_coordinates(collections)

    return collections

def get_cartesian_coordinates(collections):
    cartesian_coordinates = {}
    for collection_key in collections:
        cartesian_coordinates[collection_key] = collections[collection_key]['cartesian_coordinates']
    return cartesian_coordinates

def get_wifi_rssi(collections):
    wifi_rssi = {}
    for collection_key in collections:
        wifi_rssi[collection_key] = collections[collection_key]['wifi']
    return wifi_rssi

def show_data(collections_data, type_data='cartesian', type_plot='mds', dimension=3, path=f'images/raport3/{datetime.now()}.png', title='test', xlabel='x', ylabel='y', zlabel='z', simil_method=cosine, selection='All', nr_clusters=2):
    data = get_collection_components(collections_data, type_data)
    data_coordinates = compute_coordinates(data, type_data, type_plot, dimension, simil_method, selection)
    plot_data(data, data_coordinates, dimension, path, title, xlabel, ylabel, zlabel, nr_clusters)

def compute_coordinates(data, type_data='cartesian', type_plot='mds', dimension=3, simil_method=cosine, selection='All'):
    if type_plot == 'mds':
        similarities = compute_similarities(data, type_data, dimension, simil_method, selection)
        mds = MDS(n_components=dimension, dissimilarity='precomputed', normalized_stress="auto")
        coordinates = mds.fit_transform(similarities)
#
        print(f'Stress_value = {mds.stress_}')
        print(f'n_iter = {mds.n_iter_}')
#         print(f'params = {mds.get_metadata_routing()}')
        new_coords = []

        if dimension == 3:
            for i, collection in enumerate(data):
                new_coords.append([
                coordinates[i][0],
                coordinates[i][1],
                coordinates[i][2],
                collection[0],
                collection[len(collection)-2:]
            ])

        if dimension == 2:
            for i, collection in enumerate(data):
                new_coords.append([
                coordinates[i][0],
                coordinates[i][1],
                collection[0],
                collection[len(collection)-2:]
            ])
        return new_coords


def compute_similarities(data, type_data, dimension, simil_method, selection):
    if type_data == 'cartesian':
        return compute_cartesian_similarities(data, dimension)
    elif type_data == 'wifi':
        return compute_wifi_similarities(data, simil_method, selection)

def compute_cartesian_similarities(data, dimension):
    similarities = np.empty((len(data), len(data)))
    for i, i_key in enumerate(data):
        for j, j_key in enumerate(data):
            if dimension == 3:
                similarities[i, j] = math.dist(data[i_key][0:3],
                                              data[j_key][0:3])
            if dimension == 2:
                similarities[i, j] = math.dist(data[i_key][0:2],
                                               data[j_key][0:2])
    return similarities

def compute_wifi_similarities(data, simil_method, selection):
    # Matricea de similarități între produse
    similarities = np.empty((len(data), len(data)))
    ones = {}
    nr_ones = 0
    # TODO  sa fac completez jumatate de matrice
    for i, i_key in enumerate(data):
        for j, j_key in enumerate(data):
            similarities[i][j] = compare_locations(data[i_key], data[j_key], simil_method, selection)
            if similarities[i][j] == 1:
#                 similarities[i][j] = 0
                nr_ones += 1
                if i_key + ' ' + j_key not in ones:
                    ones[i_key + ' ' + j_key] = 0
                ones[i_key + ' ' + j_key] += 1

    print(nr_ones)
    print(ones)
    return similarities

def plot_data(data, coordinates, dimension, path, title, xlabel, ylabel, zlabel, nr_clusters):
    fig = plt.figure()
    plt.axis('equal')
    if dimension == 3:
        ax = fig.add_subplot(111, projection='3d')

    if dimension == 2:
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        for i, (x, y, nr_file, nr_ctr) in enumerate(coordinates):
            plt.scatter(x, y, color=colors[int(int(nr_file) % nr_clusters)], label=f'floor {int(nr_file) % nr_clusters}')

        for i, (x, y, nr_file, nr_ctr) in enumerate(coordinates):
            plt.text(x + .03, y + .03, nr_ctr, fontsize=9)

    if dimension == 3:
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_zlabel(zlabel)

        for i, (x, y, z, nr_file, nr_ctr) in enumerate(coordinates):
            ax.scatter(x, y, z, color=colors[int(int(nr_file) % nr_clusters)], label=f'floor {int(nr_file) % nr_clusters}')

        for i, (x, y, z, nr_file, nr_ctr) in enumerate(coordinates):
            ax.text(x + .03, y + .03, z + .03, nr_ctr, fontsize=9)

    plt.title(title)

    handles = []
    for i in range(0, nr_clusters):
        handles.append(matplotlib.patches.Patch(color=colors[int(i)], label=f'fisier{int(i)}'))

    plt.legend(handles=handles)

    plt.grid()
    plt.show()
    fig.savefig(path, bbox_inches='tight')

