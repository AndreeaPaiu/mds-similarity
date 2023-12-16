import random

from sklearn.cluster import KMeans
from sklearn.manifold import MDS

from compute_mds_wifi_similarity import *
from compute_mds_cartesian import *


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

    print(f"colectii = {len(collections)}")
    print(f"coordinates = {len(coordinates)}")
    # print(f"labels = {len(labels)}")
    # exit()
    return collections

def add_floor_collections(collections, labels):
    for i in range(len(collections)):
        collections[i]['floor'] = int(labels[i])

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


