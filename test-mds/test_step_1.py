import random
from compute_mds_wifi_similarity import *
from sklearn.manifold import MDS
import numpy as np
import copy
from numpy import linalg as LA

def compute_scale(mtx1, mtx2):
    mtx1 = np.array([rpoint1, rpoint2])
    mtx2 = np.array([point1, point2])
    media = np.mean(mtx1, axis = 0)
    mtx1 = mtx1 - media
    media = np.mean(mtx2, axis = 0)
    mtx2 = mtx2 - media
    media1 = np.mean(mtx2)
    std1 = np.std(mtx2)

    media2 = np.mean(mtx1)
    std2 = np.std(mtx1)
    print(std2 / std1)
    return std2 / std1

def start(floor1_data, floor2_data, simil_method, selection):
    floor1_data_copy = floor1_data.copy()
    random.shuffle(floor1_data_copy)
    similarities, count_floors = compute_mds_wifi_similarity(floor1_data_copy, simil_method, selection)
    mds = MDS(n_components=2, dissimilarity='precomputed')
    coordinates = mds.fit_transform(similarities)
    compute_scale(floor1_data_copy[0]['real_coordinates'], floor1_data_copy[1]['real_coordinates'], coordinates[0], coordinates[1])