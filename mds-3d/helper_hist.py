import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from compare_locations import compare_locations
from scipy.spatial.distance import cosine
import math
from helper import *
import statistics

def show_stress_histogram(collection1, type_data='wifi', path='images/raport3/test.png', title='Test', xlabel='x', ylabel='y', simil_method='', selection='All', extension='.png'):

    data = get_collection_components(collection1, type_data)

    similarities = compute_similarities(data, type_data, 2, simil_method, selection)

    coord_similarities = compute_coordinates(data, type_data, type_plot='mds', dimension=2, simil_method=cosine, selection='All')

    dist = compute_cartesian_distance(coord_similarities, dimension=2)

    stress = ((dist.ravel() - similarities.ravel()) ** 2).sum() / 2 # size(similarities.ravel())
    print(f'Stresul mediu = {stress}')
    print(f'Stress on one pair: {(stress * 2) / len(dist.ravel()) }')
    dist_array = dist.ravel()
    similarities_array = similarities.ravel()
    diff_between_points = []
    stresses = []
    for i, key in enumerate(dist_array):
        diff_between_points.append(abs(dist_array[i] - similarities_array[i]))
        stresses.append((dist_array[i] - similarities_array[i]) ** 2) # ???

    # show diff stresses
    print(f'Difference between point -> standard deviation: {statistics.stdev(diff_between_points)}' )
    print(f'Difference Stress on one pair: {diff_between_points.sum() / len(dist.ravel()) }')
    print(path)
    show_histogram(diff_between_points, 'diff ' + title, path + '_ diff' + extension, xlabel, ylabel)



    # show smacof example stress
    print(f'Smacof example stress -> standard deviation: {statistics.stdev(stresses)}' )
    show_histogram(stresses, 'smacof ' + title, path + '_smacof' + extension, xlabel, ylabel)


def show_histogram(data, title, path, xlabel, ylabel):

    fig = plt.figure()
    sns.histplot(data, bins=100, kde=True, color='skyblue')

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    plt.show()
    fig.savefig(path, bbox_inches='tight')

def compute_cartesian_distance(data, dimension):
    distances = np.empty((len(data), len(data)))
    for i, i_key in enumerate(data):
        for j, j_key in enumerate(data):
            if dimension == 3:
                distances[i, j] = math.dist(data[i][0:3],
                                              data[j][0:3])
            if dimension == 2:
                distances[i, j] = math.dist(data[i][0:2],
                                               data[j][0:2])
    return distances

def show_similarities_histogram(collection1, collection2, type_data='wifi', path='images/raport3/test.png', title='Test', xlabel='x', ylabel='y', simil_method='', selection='All'):
    if collection2 is None:
        similarities = compute_similarities_one_collection(collection1, type_data, simil_method, selection)
    else:
        similarities = compute_similarities_between_two_collections(collection1, collection2, type_data, simil_method, selection)

    fig = plt.figure()

    if type_data == 'wifi':
        sns.histplot(similarities, bins=100, kde=True, color='skyblue', binrange=(0,1))
    elif type_data == 'cartesian':
        sns.histplot(similarities, bins=100, kde=True, color='skyblue')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    plt.show()
    fig.savefig(path, bbox_inches='tight')

def compute_similarities_between_two_collections(collection1, collection2, type_data, simil_method, selection):
    if type_data == 'cartesian':
        return compute_cartesian_similarities_between_two_collections(collection1, collection2)
    elif type_data == 'wifi':
        return compute_wifi_similarities_between_two_collections(collection1, collection2, simil_method, selection)


def  compute_wifi_similarities_between_two_collections(collection1, collection2, simil_method, selection):
    similarities = []
    for key1 in collection1:
        for key2 in collection2:

            similarities.append(compare_locations(collection1[key1]['wifi'], collection2[key2]['wifi'], simil_method, selection))
    return similarities

def compute_cartesian_similarities_between_two_collections(collection1, collection2):
    similarities = []
    for key1 in collection1:
        for key2 in collection2:
            similarities.append(math.dist(collection1[key1]['cartesian_coordinates'][0:3],collection2[key2]['cartesian_coordinates'][0:3]))
    return similarities

def compute_similarities_one_collection(collection, type_data, simil_method, selection):
    if type_data == 'cartesian':
        return compute_cartesian_similarities_one_collection(collection)
    elif type_data == 'wifi':
        return compute_wifi_similarities_one_collection(collection, simil_method, selection)


def  compute_wifi_similarities_one_collection(collection, simil_method, selection):
    similarities = []

    for i, key1 in enumerate(collection):
        for j, key2 in enumerate(collection):
            if i > j:
                similarities.append(compare_locations(collection[key1]['wifi'], collection[key2]['wifi'], simil_method, selection))
    return similarities

def compute_cartesian_similarities_one_collection(collection):
    similarities = []
    for i, key1 in enumerate(collection):
        for j, key2 in enumerate(collection):
            if i > j:
                similarities.append(math.dist(collection[key1]['cartesian_coordinates'][0:3],collection[key2]['cartesian_coordinates'][0:3]))

    return similarities
