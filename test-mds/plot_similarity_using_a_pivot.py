import math

import matplotlib.pyplot as plt
from scipy.stats import cosine

from andreea_test.compare_locations import compare_locations

def real_distantce(c1, c2):
    return math.sqrt(math.pow(c2[0] - c1[0], 2) + math.pow(c2[1] - c1[1], 2))

def plot_similarity_using_a_pivot(collections):
    similarity_braycurtis = []
    # similarity_cosine = []
    real_distance = []
    for collection in collections:
        if 'wifi' not in collection:
            continue
        similarity_braycurtis.append(compare_locations(collections[0], collection))
        # similarity_cosine.append(compare_locations(collections[0], collection, selection ='Average', dif=True, simil_method=cosine))

        if 'real_coordinates' not in collection:
            continue
        real_distance.append(real_distantce(collections[0]['real_coordinates'], collection['real_coordinates']))

    for i in range(len(similarity_braycurtis)):
        plt.plot(real_distance[i], similarity_braycurtis[i], 'o', color='blue')

    # for i in range(len(similarity_cosine)):
    #     plt.plot(real_distance[i], similarity_braycurtis[i], 'o', color='red')

    plt.title('Similarity using a pivot')
    plt.xlabel('distance(m)')
    plt.ylabel('similarity')
    plt.show()
