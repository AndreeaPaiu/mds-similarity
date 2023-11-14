import numpy as np
from matplotlib import pyplot as plt


def plot_similarity_between_points(points):
    fig = plt.figure()
    m = np.polyfit([r[0] for r in points], [r[1] for r in points], 2)
    print(f"Slope = {m}")
    predict = np.poly1d(m)
    x_lin_reg = range(0, 8)
    y_lin_reg = predict(x_lin_reg)
    plt.plot(x_lin_reg, y_lin_reg, c='b', alpha=0.2, linewidth=3)

    plt.plot([r[0] for r in points], [r[1] for r in points], ".", label='bc', alpha=0.2)
    plt.xlabel('distance[m]')
    plt.ylabel('dissimilarity')
    plt.grid(True)
    plt.title(f"Dataset 1b pairs={len(points)}")
    # plt.xticks(range(0, 20, 2))
    # plt.ylim(0, 1.0)
    plt.xlim(0, 7.0)
    # plt.yticks(np.arange(0, 1, 0.1))
    # plt.legend(loc="upper left",shadow=True, fancybox=True)
    plt.show()
    fig.savefig(f"braycurtis-vs-distance-ds1b.svg", bbox_inches='tight')