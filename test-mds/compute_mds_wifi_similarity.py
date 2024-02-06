import numpy as np
from compare_locations import compare_locations


def compute_mds_wifi_similarity(collections, simil_method, selection):
    # Matricea de similarități între produse
    similarities = np.empty((len(collections), len(collections)))
    count_floors = {}
    sum_similarity = 0
    count_similarity = 0

    for i in range(len(collections)):
        if 'floor' in collections[i]:
            if int(collections[i]['floor']) not in count_floors:
                count_floors[int(collections[i]['floor'])] = 0
            count_floors[int(collections[i]['floor'])] += 1
        for j in range(len(collections)):
            if 'wifi' not in collections[i] or 'wifi' not in collections[j]:
                similarities[i][j] = 1
                continue

            similarities[i][j] = compare_locations(collections[i], collections[j], simil_method, selection=selection)

            # if similarities[i][j] > 0.9:
            #     print('here')
            #     similarities[i][j] = 0
            # if similarities[i][j] < 1:
            #     if similarities[i][j] < 0.8:
            #         sum_similarity += similarities[i][j]
            #         count_similarity += 1
            #     if similarities[i][j] < 0.7:
            #         sum_similarity += similarities[i][j]
            #         count_similarity += 1
            #     if similarities[i][j] < 0.6:
            #         sum_similarity += similarities[i][j]
            #         count_similarity += 1
            #     if similarities[i][j] < 0.5:
            #         sum_similarity += similarities[i][j]
            #         count_similarity += 1
            #     if similarities[i][j] < 0.4:
            #         sum_similarity += similarities[i][j]
            #         count_similarity += 1
            #     if similarities[i][j] < 0.3:
            #         sum_similarity += similarities[i][j]
            #         count_similarity += 1
            #     if similarities[i][j] < 0.2:
            #         sum_similarity += similarities[i][j]
            #         count_similarity += 1
            #     if similarities[i][j] < 0.1:
            #         sum_similarity += similarities[i][j]
            #         count_similarity += 1
            #     sum_similarity += similarities[i][j]
            #     count_similarity += 1

            if 'floor' in collections[i] and 'floor' in collections[j]:
                similarities[i][j] += abs(int(collections[j]['floor']) - int(collections[i]['floor']))

    # avg_similarities = sum_similarity / count_similarity
    # print(avg_similarities)
    #
    # for i in similarities:
    #     for j in range(len(i)):
    #         if i[j] > avg_similarities:
    #             i[j] = 1

    return similarities, count_floors
