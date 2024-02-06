import numpy as np
from matplotlib import pyplot as plt
from compute_similarities_using_neighbors import *


def plot_similarity_between_points(collections, aps=[], selection='All', simil_method=braycurtis, title='', file_name='images/test.svg', sm=compute_similarities_using_neighbors):
    bd = compute_similarities_using_neighbors(
        collections,
        aps=aps,
        selection=selection,
        simil_method=simil_method
    )
    points = [[r[2], r[3]] for r in bd]
    fig = plt.figure()
    m = np.polyfit([r[0] for r in points], [r[1] for r in points], 2)
    print(f"Slope = {m}")
    predict = np.poly1d(m)
    x_lin_reg = range(0, 20)
    y_lin_reg = predict(x_lin_reg)
    plt.plot(x_lin_reg, y_lin_reg, c='b', alpha=0.2, linewidth=3)

    plt.plot([r[0] for r in points], [r[1] for r in points], ".", label='bc', alpha=0.2)
    plt.xlabel('distance[m]')
    plt.ylabel('dissimilarity')
    plt.grid(True)
    plt.title(title)
    # plt.xticks(range(0, 20, 2))
    plt.ylim(0, 1.0)
    # plt.xlim(0, 20.0)
    plt.xlim(0, 12.0)
    # plt.yticks(np.arange(0, 1, 0.1))
    # plt.legend(loc="upper left",shadow=True, fancybox=True)
    plt.show()
    fig.savefig(file_name, bbox_inches='tight')

def plot_all_similarity_between_points(collections, simil_methods, selections, aps=[], sm=compute_similarities_using_neighbors):
    for simil_method in simil_methods:
        for selection in selections:
            plot_similarity_between_points(
                collections,
                aps=aps,
                selection=selection,  # Comm | All
                simil_method=simil_method,
                title=f"Dataset 1b {simil_method.__name__} vs distance with {selection} aps",
                file_name=f"images/{simil_method.__name__}-vs-distance-ds1-with-{selection}-aps.png"
            )