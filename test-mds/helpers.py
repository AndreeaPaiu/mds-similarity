import random

import matplotlib
import numpy
import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial import procrustes
from sklearn.cluster import KMeans
from sklearn.manifold import MDS

from compute_mds_wifi_similarity import *
from compute_mds_cartesian import *
from scipy.spatial.distance import cosine
from sklearn.decomposition import PCA
from scipy.linalg import orthogonal_procrustes

colors = []
for color in matplotlib.colors.TABLEAU_COLORS:
    colors.append(color)


def add_floor_label_to_collection(collections, similarities, n_clusters, n_dim=3):
    # Crearea unui obiect MDS cu n dimensiuni
    # metric = False => valorile de 0 sunt cosidetare valori lipsa
    mds = MDS(n_components=n_dim, dissimilarity='precomputed')

    # Aplicarea MDS pe matricea de similarități
    coordinates = mds.fit_transform(similarities)

    # clasificare puncte in 2 etaje
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(coordinates)

    add_floor_collections(collections, kmeans.labels_)

    return collections


def add_floor_collections(collections, labels):
    for i in range(len(collections)):
        collections[i]['floor'] = int(labels[i])


def order_floor_collection(collections):
    # separ pe etaje
    floors_collections = {}
    for collection in collections:
        if collection['floor'] not in floors_collections:
            floors_collections[collection['floor']] = []
        floors_collections[collection['floor']].append(collection)

    # iau random un punct de plecare (de exemplu primul, oricum sunt amestecate) si ii caut cel mai apropiat punct
    # printre toate etajele inafara de etajul curent.
    first_point = collections[0]
    print(first_point)
    exit()


def compute_2D_mds(collections, simil_method, ref_coordinates=None, selection='All', type_data='wifi', ref_collection=None, first_collection=False, scale=23):
    floor_id = int(collections[0]['floor_id'])
    similarities = []
    mtx = []

    # Matricea de similarități între puncte( cate un etaj)
    if type_data == 'wifi':
        similarities, count_floors = compute_mds_wifi_similarity(collections, simil_method, selection)

    if type_data == 'cartesian':
        similarities, count_floors = compute_mds_cartesian(collections, 2)

    # Crearea unui obiect MDS cu n dimensiuni
    mds = MDS(n_components=2, dissimilarity='precomputed')

    # Aplicarea MDS pe matricea de similarități
    coordinates = mds.fit_transform(similarities)
    print(first_collection)
    print(ref_coordinates)
    print(ref_collection)

    if not first_collection:
        mtx_r1 = np.array([ref_coordinates[ref_collection[0][0]][0:2], ref_coordinates[ref_collection[1][0]][0:2], ref_coordinates[ref_collection[2][0]][0:2], ref_coordinates[ref_collection[3][0]][0:2]])
        mtx_r2 = np.array([coordinates[ref_collection[0][1]], coordinates[ref_collection[1][1]], coordinates[ref_collection[2][1]], coordinates[ref_collection[3][1]]])

        # centrat in (0,0)
        media = np.mean(mtx_r1, axis = 0)
        mtx_r1 = mtx_r1 - media

        rotation, _ = orthogonal_procrustes(mtx_r1, mtx_r2)
    else:
        mtx_r1 = np.array([collections[0]['real_coordinates'][0:2], collections[12]['real_coordinates'][0:2], collections[25]['real_coordinates'][0:2], collections[37]['real_coordinates'][0:2]])
        mtx_r2 = np.array([coordinates[0], coordinates[12], coordinates[25], coordinates[37]])

        # centrat in (0,0)
        media = np.mean(mtx_r1, axis = 0)
        mtx_r1 = mtx_r1 - media

        rotation, _ = orthogonal_procrustes(mtx_r1, mtx_r2)

    coordinates = coordinates@rotation.T
    coordinates = coordinates * scale
    # align points
    # align_points_coordinates = []
    # if ref_coordinates is not None and ref_collection is not None:
    #     mtx, align_points_coordinates, disparity = align_points(ref_collection, collections, ref_coordinates, coordinates)

    new_coords = []
    for i in range(len(coordinates)):
        new_coords.append([
            coordinates[i][0],
            coordinates[i][1],
            floor_id,
            collections[i]['label_id']
        ])

    # return new_coords, coordinates, mtx
    return new_coords

def align_points(ref_collection, source_collection, ref_points, source_points):

    # compute_correspondences
    correspondences = mapping_floors_nearst_point_distance(ref_collection, source_collection)

    points_reordered = np.array([source_points[idx] for idx in [corr[1] for corr in correspondences]])

    return procrustes(ref_points, points_reordered)

def find_nearest_point_similarity(point, collections, simil_method=cosine, selection="All"):
    # random.shuffle(collections)
    similarity = 1.0  # stiu ca cea mai mare similaritate e 1, si as vrea sa nu ma pronunt la aceasta valoare
    nearestpoint = {}
    for collection in collections:
        if 'wifi' not in point or 'wifi' not in collection:
            continue

        new_similarity = compare_locations(point, collection, simil_method, selection=selection)

        if similarity > new_similarity:
            nearestpoint = collection.copy()
            similarity = new_similarity

    return nearestpoint


def mapping_floors_nearst_point_similarity(collections1, collections2, simil_method=cosine, selection="All"):
    # random.shuffle(collections2)
    similarities = {}
    for collection_1 in collections1:
        print(collection_1['label_id'])
        similarities[collection_1['label_id']] = \
        find_nearest_point_similarity(collection_1, collections2, simil_method, selection)['label_id']
    print(similarities)

    similarities_set = set()
    for key in similarities:
        similarities_set.add(similarities[key])
        print(f"{key} -> {similarities[key]}")

    print(similarities_set)
    print(f"numarul de puncte vazute ca cele mai apropiate {len(similarities_set)}")
    exit()


# distanta folosit coordonatele de la mds
def find_nearest_point_distance(point, collections, used=[]):
    # random.shuffle(collections)
    distance = 1000
    nearest_point = -1

    for collection in collections:
        if collection['label_id'] in used:
            continue
        new_distance = math.dist(point, collection['coord'])
        # new_distance = math.sqrt(math.pow(collection['coord'][0] - point[0], 2) +
        #         math.pow(collection['coord'][1] - point[1], 2) +
        #         math.pow(collection['coord'][2] - point[2], 2) * 1.0)
        if distance > new_distance:
            nearest_point = collection['label_id']
            distance = new_distance

    return nearest_point


def mapping_floors_nearst_point_distance(collections1, collections2, simil_method=cosine, selection="All"):
    merge_data = collections1 + collections2
    similarities, count_floors = compute_mds_wifi_similarity(merge_data, simil_method, selection)
    mds = MDS(n_components=3, dissimilarity='precomputed')

    # Aplicarea MDS pe matricea de similarități
    coordinates = mds.fit_transform(similarities)
    pca = PCA(3)
    pca.fit(coordinates)
    coordinates = pca.fit_transform(coordinates)

    for i in range(len(merge_data)):
        merge_data[i]['coord'] = [coordinates[i][1], coordinates[i][2]]

    distances = {}
    used = []
    used_etp = []
    unfounded = []
    unfounded_etp = collections1

    while True:
        if len(collections1) < len(collections2) and len(unfounded_etp) == 0:
            break
        elif len(collections1) >= len(collections2) and len(unfounded_etp) == abs(len(collections1) - len(collections2)):
            break
        used = list(set(used + used_etp))
        for i in range(len(unfounded_etp)):
            # get the nearest point
            distances[unfounded_etp[i]['label_id']] = find_nearest_point_distance(unfounded_etp[i]['coord'],
                                                                               merge_data[len(collections1):],
                                                                               used)

            # check if points is used
            if distances[unfounded_etp[i]['label_id']] in used_etp:
                unfounded.append(unfounded_etp[i])

            # point is used
            used_etp.append(distances[unfounded_etp[i]['label_id']])

        unfounded_etp = unfounded
        unfounded = []

    # distances_set = set()
    distances_array = []
    # for key in distances:
    #     distances_set.add(distances[key])
    #     print(f"{key} -> {distances[key]}")
    for i in range(len(distances)):
        distances_array.append((i, distances[i]))

    # print(f"numarul de puncte vazute ca cele mai apropiate {len(distances_set)}")

    lines_coords = []

    # for i in range(len(collections1)):
    # line = {}
    # line['0'] = coordinates[i]
    # line['1'] = distances[i]['coord']
    # lines_coords.append(line)
    # ploting_3D(coordinates, merge_data)
    print(distances_array)
    return distances_array


def ploting_3D(coordinates, collections):
    fig = plt.figure()
    plt.axis('equal')
    ax = fig.add_subplot(111, projection='3d')
    plt.xlabel('x')
    plt.ylabel('y')
    for i, (x, y, z) in enumerate(coordinates):
        ax.scatter(x, y, z, color=colors[int(collections[i]['floor_id'])], label=f'Line {i + 1}')
        ax.text(x + .03, y + .03, z + .03, collections[i]['label_id'], fontsize=9)

    plt.grid()
    plt.show()


def ploting_2D(coordinates, collections):
    fig = plt.figure()
    plt.axis('equal')
    for i, (x, y) in enumerate(coordinates):
        plt.scatter(x, y, color=colors[int(collections[i]['floor_id'])], label=f'Line {i + 1}')
        plt.text(x + .03, y + .03, collections[i]['label_id'], fontsize=9)

    plt.grid()
    plt.show()


# cartesian system
def find_nearest_point_distance_coord(point, collections):
    distance = 10000000
    nearestpoint = {}

    for collection in collections:
        new_distance = math.dist(point['real_coordinates'], collection['real_coordinates'])
        if distance > new_distance:
            nearestpoint = collection
            distance = new_distance

    return nearestpoint


def mapping_floors_nearst_point_coord(collections1, collections2):
    distances = {}
    for collection_1 in collections1:
        distances[collection_1['label_id']] = find_nearest_point_distance_coord(collection_1, collections2)['label_id']
    distances_set = set()
    for key in distances:
        distances_set.add(distances[key])
        print(f"{key} -> {distances[key]}")

    print(f"numarul de puncte vazute ca cele mai apropiate {len(distances_set)}")


def f_add_noise(collections, range_value=0.5):
    for collection in collections:
        collection['real_coordinates'][0] += random.uniform(-range_value, range_value)
        collection['real_coordinates'][1] += random.uniform(-range_value, range_value)
        collection['real_coordinates'][2] += random.uniform(-range_value, range_value)
    return collections


# folosesc cate un punct din fiecare etaj pentru a sorta punctele
# punctul 0, care sunt considerate cele mai apropiate
# nu pot tine cont de directie, adica daca e primul sau ultimul etaj
def order_floors_using_only_one_points_per_floor(points):
    similarities, count_floors = compute_mds_wifi_similarity(points, cosine, 'All')
    similarities_2 = similarities.copy()
    distances = [points[0]['floor_id']]
    i = 0
    while 1:
        similarities_2[i][i] = 2.
        min_value = numpy.min(similarities_2[i])
        min_index = numpy.where(np.isclose(similarities_2[i], min_value))[0][0]
        x = numpy.where(distances == points[min_index]['floor_id'])
        if len(x[0]) == 0:
            distances = numpy.append(distances, points[min_index]['floor_id'])
            i = min_index
            similarities_2[i][min_index] = 2.
        else:
            similarities_2[i][min_index] = 2.
            min_value = numpy.min(similarities_2[i])
            min_index = numpy.where(np.isclose(similarities_2[i], min_value))[0][0]
            x = numpy.where(distances == points[min_index]['floor_id'])
            if len(x[0]) == 0:
                distances = numpy.append(distances, points[min_index]['floor_id'])
                i = min_index
                similarities_2[i][min_index] = 2.
            else:
                break
    i = 1
    while 1:
        min_value = numpy.min(similarities_2[i])
        min_index = numpy.where(np.isclose(similarities_2[i], min_value))[0][0]
        x = numpy.where(distances == points[min_index]['floor_id'])
        if len(x[0]) == 0:
            distances = numpy.insert(distances, 0, points[min_index]['floor_id'])
            i = min_index
            print(f'{points[min_index]["floor_id"]} -> {min_index} ->{min_value}')
            similarities_2[i][min_index] = 2.
        else:
            similarities_2[i][min_index] = 2.
            min_value = numpy.min(similarities_2[i])
            min_index = numpy.where(np.isclose(similarities_2[i], min_value))[0][0]
            x = numpy.where(distances == points[min_index]['floor_id'])
            if len(x[0]) == 0:
                distances = numpy.insert(distances, 0, points[min_index]['floor_id'])
                i = min_index
                print(f'{points[min_index]["floor_id"]} -> {min_index} ->{min_value}')
                similarities_2[i][min_index] = 2.
            else:
                break

    print(distances)
    return distances


def order_floors_using_all_points(points, similarities):
    similarities_2 = similarities.copy()

    i = 0
    distances = [points[i]['floor']]

    while 1:
        x = [index for index in range(len(points)) if points[index]['floor'] == points[i]['floor']]
        for index in x:
            similarities_2[i][index] = 2.
        min_value = numpy.min(similarities_2[i])
        min_index = numpy.where(np.isclose(similarities_2[i], min_value))[0][0]
        x = numpy.where(distances == points[min_index]['floor'])
        if len(x[0]) == 0:
            distances = numpy.append(distances, points[min_index]['floor'])
            x = [index for index in range(len(points)) if points[index]['floor'] == points[min_index]['floor']]
            for index in x:
                similarities_2[i][index] = 2.
            i = min_index
        else:
            x = [index for index in range(len(points)) if points[index]['floor'] == points[min_index]['floor']]
            for index in x:
                similarities_2[i][index] = 2.
            min_value = numpy.min(similarities_2[i])
            min_index = numpy.where(np.isclose(similarities_2[i], min_value))[0][0]
            x = numpy.where(distances == points[min_index]['floor'])
            if len(x[0]) == 0:
                distances = numpy.append(distances, points[min_index]['floor'])
                similarities_2[i][min_index] = 2.
                i = min_index
            else:
                break
    # i = 0
    # while 1:
    #     x = [index for index in range(len(points)) if points[index]['floor_id'] == points[i]['floor_id']]
    #     for index in x:
    #         similarities_2[i][index] = 2.
    #     min_value = numpy.min(similarities_2[i])
    #     min_index = numpy.where(np.isclose(similarities_2[i], min_value))[0][0]
    #     x = numpy.where(distances == points[min_index]['floor_id'])
    #     if len(x[0]) == 0:
    #         distances = numpy.insert(distances, 0, points[min_index]['floor_id'])
    #         x = [index for index in range(len(points)) if points[index]['floor_id'] == points[min_index]['floor_id']]
    #         for index in x:
    #             similarities_2[i][index] = 2.
    #         i = min_index
    #     else:
    #         similarities_2[i][min_index] = 2.
    #         min_value = numpy.min(similarities_2[i])
    #         min_index = numpy.where(np.isclose(similarities_2[i], min_value))[0][0]
    #         x = numpy.where(distances == points[min_index]['floor_id'])
    #         if len(x[0]) == 0:
    #             distances = numpy.insert(distances, 0, points[min_index]['floor_id'])
    #             x = [index for index in range(len(points)) if
    #                  points[index]['floor_id'] == points[min_index]['floor_id']]
    #             for index in x:
    #                 similarities_2[i][index] = 2.
    #             i = min_index
    #         else:
    #             break

    return distances


def set_right_floor_id(points, order):
    for i in range(len(order)):
        x = [index for index in range(len(points)) if points[index]['floor'] == order[i]]
        for index in x:
            points[index]['floor_id'] = i

def compute_rotation_and_scale(collection, simil_method=cosine, selection='All'):
    # Matricea de similarități între puncte( cate un etaj)
    similarities, count_floors = compute_mds_wifi_similarity(collection, simil_method, selection)

    # Crearea unui obiect MDS cu n dimensiuni
    mds = MDS(n_components=2, dissimilarity='precomputed')

    # Aplicarea MDS pe matricea de similarități
    coordinates = mds.fit_transform(similarities)

    mtx_r1 = np.array([collection[0]['real_coordinates'][0:2], collection[12]['real_coordinates'][0:2], collection[25]['real_coordinates'][0:2], collection[37]['real_coordinates'][0:2]])
    mtx_r2 = np.array([coordinates[0], coordinates[12], coordinates[25], coordinates[37]])

    # centrat in (0,0)
    media = np.mean(mtx_r1, axis = 0)
    mtx_r1 = mtx_r1 - media

    R, s = orthogonal_procrustes(mtx_r1, mtx_r2)

    std1 = np.std(mtx_r2)
    std2 = np.std(mtx_r1)

    return R, std2 / std1
