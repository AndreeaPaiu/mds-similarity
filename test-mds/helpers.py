import random

import matplotlib
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.manifold import MDS

from compute_mds_wifi_similarity import *
from compute_mds_cartesian import *
from scipy.spatial.distance import cosine

colors = []
for color in matplotlib.colors.TABLEAU_COLORS:
    colors.append(color)


def add_floor_label_to_collection(collections, simil_method, n_clusters, selection='All', type_data='wifi', n_dim=3):
    similarities = []

    # Matricea de similarități între produse
    if type_data == 'wifi':
        similarities, count_floors = compute_mds_wifi_similarity(collections, simil_method, selection)

    if type_data == 'cartesian':
        similarities, count_floors = compute_mds_cartesian(collections, n_dim)

    # Crearea unui obiect MDS cu n dimensiuni
    # metric = False => valorile de 0 sunt cosidetare valori lipsa
    mds = MDS(n_components=n_dim, dissimilarity='precomputed')

    # Aplicarea MDS pe matricea de similarități
    coordinates = mds.fit_transform(similarities)

    # clasificare puncte in 2 etaje
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(coordinates)

    add_floor_collections(collections, kmeans.labels_)

    # print(f"labels = {len(labels)}")
    # exit()
    return collections

def add_floor_collections(collections, labels):
    for i in range(len(collections)):
        collections[i]['floor'] = int(labels[i])

def order_floor_collection(collections):
    #separ pe etaje
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




def compute_2D_mds(collections, simil_method, selection='All', type_data='wifi'):

    floor_number = int(collections[0]['floor'])
    floor_id = int(collections[0]['floor_id'])
    similarities = []

    # Matricea de similarități între produse
    if type_data == 'wifi':
        similarities, count_floors = compute_mds_wifi_similarity(collections, simil_method, selection)

    if type_data == 'cartesian':
        similarities, count_floors = compute_mds_cartesian(collections, 2)

    # Crearea unui obiect MDS cu n dimensiuni
    mds = MDS(n_components=2, dissimilarity='precomputed')

    # Aplicarea MDS pe matricea de similarități

    coordinates = mds.fit_transform(similarities)


    # # Aflarea numărului de coloane
    # num_cols = len(coordinates[1])
    #
    # sums = [sum(row[i] for row in coordinates) for i in range(num_cols)]
    # x = sums[0] / len(coordinates)
    # y = sums[1] / len(coordinates)


    new_coords = []
    for coord in coordinates:
        new_coords.append([
            coord[0],
            coord[1],
            floor_number,
            floor_id
        ])

    return new_coords

def find_nearest_point_similarity(point, collections, simil_method=cosine, selection="All"):
    # random.shuffle(collections)
    similarity = 1.0  # stiu ca cea mai mare similaritate e 1, si as vrea sa nu ma pronunt la aceasta valoare
    nearestpoint = {}
    for collection in collections:
        if 'wifi' not in point or 'wifi' not in collection:
            continue

        new_similarity = compare_locations(point, collection, simil_method, selection=selection)

        if similarity > new_similarity:
            nearestpoint = collection
            similarity = new_similarity

    return nearestpoint

def mapping_floors_nearst_point_similarity(collections1, collections2, simil_method=cosine, selection="All"):
    # random.shuffle(collections2)
    similarities = {}
    for collection_1 in collections1:
        similarities[collection_1['label_id']] = find_nearest_point_similarity(collection_1, collections2, simil_method=cosine, selection="All")['label_id']

    similarities_set = set()
    for key in similarities:
        similarities_set.add(similarities[key])
        print(f"{key} -> {similarities[key]}")
    print(f"numarul de puncte vazute ca cele mai apropiate {len(similarities_set)}")


# distanta folosit coordonatele de la mds
def find_nearest_point_distance(point, collections):
    # random.shuffle(collections)
    distance = 10
    nearestpoint = {}

    for collection in collections:

        new_distance = math.dist(point, collection['coord'])
        # new_distance = math.sqrt(math.pow(collection['coord'][0] - point[0], 2) +
        #         math.pow(collection['coord'][1] - point[1], 2) +
        #         math.pow(collection['coord'][2] - point[2], 2) * 1.0)
        if distance > new_distance:
            nearestpoint = collection
            distance = new_distance

    return nearestpoint

def mapping_floors_nearst_point_distance(collections1, collections2 , simil_method=cosine, selection="All"):
    merge_data = collections1 + collections2
    similarities, count_floors = compute_mds_wifi_similarity(merge_data, simil_method, selection)
    mds = MDS(n_components=3, dissimilarity='precomputed')

    # Aplicarea MDS pe matricea de similarități

    coordinates = mds.fit_transform(similarities)

    for i in range(len(coordinates)):
        merge_data[i]['coord'] = [coordinates[i][0], coordinates[i][1], int(merge_data[i]['floor_id'])]
        # merge_data[i]['coord'] = coordinates[i]
    distances = {}
    for i in range(len(collections1)):
        distances[merge_data[i]['label_id']] = find_nearest_point_distance(merge_data[i]['coord'], merge_data[len(collections1):])

    distances_set = set()
    for key in distances:
        distances_set.add(distances[key]['label_id'])
        print(f"{key} -> {distances[key]['label_id']}")

    print(f"numarul de puncte vazute ca cele mai apropiate {len(distances_set)}")

    print(distances)
    exit()
    lines_coords = []

    for i in range(len(collections1)):
        line = {}
        line['0'] = coordinates[i]
        line['1'] = distances[i]['coord']
        lines_coords.append(line)
    ploting_3D(coordinates, merge_data)

def ploting_3D(coordinates,collections):
    fig = plt.figure()
    plt.axis('equal')
    ax = fig.add_subplot(111, projection='3d')
    for i, (x, y, z) in enumerate(coordinates):
        ax.scatter(x, y, z, color=colors[int(collections[i]['floor_id'])], label=f'Line {i + 1}')
        ax.text(x + .03, y + .03, z + .03, collections[i]['label_id'], fontsize=9)

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



def f_add_noise(collections, range_value = 0.5):
    for collection in collections:
        collection['real_coordinates'][0] += random.uniform(-range_value, range_value)
        collection['real_coordinates'][1] += random.uniform(-range_value, range_value)
        collection['real_coordinates'][2] += random.uniform(-range_value, range_value)
    return collections