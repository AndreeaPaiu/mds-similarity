import numpy as np
from compare_locations import compare_locations


def compute_mds_wifi_similarity(collections, simil_method, selection, aps):
    # Matricea de similarități între produse
    similarities = np.empty((len(collections), len(collections)))
    count_floors = {}

    for i in range(len(collections)):
        if int(collections[i]['floor']) not in count_floors:
            count_floors[int(collections[i]['floor'])] = 0
        count_floors[int(collections[i]['floor'])] += 1
        for j in range(len(collections)):
            if 'wifi' not in collections[i] or 'wifi' not in collections[j]:
                similarities[i][j] = 1
                continue

            similarities[i][j] = compare_locations(collections[i], collections[j], simil_method, selection=selection,
                                                   all_aps=aps)
    return similarities, count_floors
