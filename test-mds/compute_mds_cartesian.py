import math

import numpy as np


def compute_mds_cartesian(collections, n_dim):
    # Matricea de similarități între produse
    similarities = np.empty((len(collections), len(collections)))
    count_floors = {}

    for i in range(len(collections)):
        if int(collections[i]['floor_id']) not in count_floors:
            count_floors[int(collections[i]['floor_id'])] = 0
        count_floors[int(collections[i]['floor_id'])] += 1
        for j in range(len(collections)):
            if 'real_coordinates' not in collections[i] or 'real_coordinates' not in collections[j]:
                similarities[i][j] = 100
                continue

            if n_dim == 3:
                similarities[i, j] = math.dist(collections[i]['real_coordinates'][0:3],
                                               collections[j]['real_coordinates'][0:3])
            if n_dim == 2:
                similarities[i, j] = math.dist(collections[i]['real_coordinates'][0:2],
                                               collections[j]['real_coordinates'][0:2])
    return similarities, count_floors