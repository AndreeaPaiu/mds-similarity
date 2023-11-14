import numpy as np
import scipy
import seaborn
from matplotlib import pyplot as plt
from scipy.spatial.distance import braycurtis, euclidean

from compare_locations import compare_locations
import scipy.stats as ss
import seaborn as sns



def test_queryvsall(query, collections, selection='Average'):
    """
        query = collection, an array or fingerprints
        collections = aray of collection, can be a floor
        RETURNS (similarity, index, eu_distance)
        """
    min = 1000.0
    mini = -1
    for c in range(0, len(collections)):
        d = compare_locations(query, collections[c], braycurtis, dif=True)
        if d != 0 and d < min:
            min = d
            mini = c
    return min, mini, \
           euclidean([query['real_coordinates'][0], query['real_coordinates'][1], query['real_coordinates'][2]],
                     [collections[mini]['real_coordinates'][0], collections[mini]['real_coordinates'][1], collections[mini]['real_coordinates'][2]])

def plot_similarity_to_nearby_point(phone_data_1, phone_data_2):
    cl_sim = []
    cl_steps = []
    cl_dist = []
    every = 3  # place documents every 6 steps (one step = 0.6m)
    query = [np.roll(phone_data_1, 0)[0::every]]
    db = [np.roll(phone_data_2, 0)[0::every]]
    for eq in range(0, len(query)):
        for q in range(0, len(query[eq])):
            sim, index, d = test_queryvsall(query[eq][q], db[eq], 'Average')
            # print(q, query[eq][q]['real_coordinates'][0], query[eq][q]['real_coordinates'][1], sim, diff_modulo(q, index, len(phone_data_1)), d)
            cl_sim.append(sim)
            # cl_steps.append(diff_modulo(p, index, len(petaje[e])))
            cl_dist.append(d)

    # query = [np.roll(phone_data_2, 0)[0::every]]
    # db = [np.roll(phone_data_1, 0)[0::every]]
    # # for e in len(retaje):
    # for eq in range(0, len(query)):
    #     for q in range(0, len(query[eq])):
    #         sim, index, d = test_queryvsall(query[eq][q], db[eq], 'Average')
    #         print(q, query[eq][q]['real_coordinates'][0], query[eq][q]['real_coordinates'][1], sim, diff_modulo(q, index, len(phone_data_1)), d)
    #         cl_sim.append(sim)
    #         # cl_steps.append(diff_modulo(p, index, len(petaje[e])))
    #         cl_dist.append(d)

    # print(np.percentile(cl_dist, [50, 95, 99]))

    b = np.arange(0, 0.4, 0.01)  # Bins of histogram - from 1 to 5
    bin_width = b[1] - b[0]
    hist_sim, bins_out = np.histogram(np.array(cl_sim), b, density=False)
    hist_sim = hist_sim / len(cl_sim)
    fig = plt.figure(figsize=(4, 3))
    plt.bar(b[:-1], hist_sim, width=bin_width, ec='k', alpha=0.8)
    plt.xlabel("Dissimilarity")
    plt.ylabel("PMF")
    plt.title("{}{}".format("Dissimilarity at the same point (Dataset 1b)", " "))
    plt.tight_layout()
    plt.show()
    fig.savefig(f"disim-same-dataset1b.pdf", bbox_inches='tight')
    fig.savefig(f"disim-same-dataset1b.svg", bbox_inches='tight')
    #    print("Percentages: ", hist_away)

    bd = np.arange(0, 10, 1)  # Bins of histogram - from 1 to 5
    bin_width = bd[1] - bd[0]
    hist_dist, bins_out = np.histogram(np.array(cl_dist), bd, density=False)
    hist_dist = hist_dist / len(cl_dist)
    fig = plt.figure(figsize=(4, 3))
    plt.bar(bd[:-1], hist_dist, width=bin_width, ec='k', alpha=0.8)
    plt.xlabel("closest point (meters)")
    plt.ylabel("PMF")
    plt.title("{} {} {}".format("Cross validation Dataset 1a ", len(hist_dist), " points;"))
    plt.tight_layout()
    plt.show()
    fig.savefig(f"dist-closest-cross-dataset1b.pdf", bbox_inches='tight')
    fig.savefig(f"dist-closest-cross-dataset1b.svg", bbox_inches='tight')
    #    print("Percentages: ", hist_away)

    cdf = np.cumsum(hist_sim)
    fig = plt.figure(figsize=(4, 3))
    plt.axhline(y=1, color='r', linestyle='--')
    plt.xlabel("Dissimilarity")
    plt.ylabel("CDF(%)")
    plt.title("{}{}".format("Dissimilarity at the same point (Dataset 1b)", ""))
    plt.plot(b[:-1], cdf)
    plt.grid(True)
    plt.tight_layout()
    plt.text(0.2, 0.4, f'99% = {np.percentile(cl_sim, [99])[0]:.2f}\n99.5% = {np.percentile(cl_sim, [99.5])[0]:.2f}')
    plt.show()
    fig.savefig(f"disim-same-cdf-dataset1b.pdf", bbox_inches='tight')
    fig.savefig(f"disim-same-cdf-dataset1b.svg", bbox_inches='tight')

    print(np.percentile(cl_sim, [50, 95, 99, 99.5]))
