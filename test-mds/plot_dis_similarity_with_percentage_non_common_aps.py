import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial.distance import braycurtis

from compare_locations import compare_locations


def plot_dis_similarity_with_percentage_non_common_aps(c1, c2, dif=True, simil_method=braycurtis):
    q, p = compare_locations(c1, c2, dif=True, simil_method=braycurtis, return_type='rssi')
    q, p = q[0:20], p[0:20]
    l = len(q)
    bc_curve = dict()
    print(l)

    for pow in [-55, -65, -75, -85]:
        bcx = []
        for i in range(0, l):
            bcx.append(braycurtis(list(q) + list((-95.0) * np.ones(i)),
                                  list(p) + list(1.0 * (pow) * np.ones(i))))
        bc_curve[pow] = np.arange(0, 1, 1 / l), bcx

    fig = plt.figure(figsize=(4, 3))  # bc, realbc, iou

    for pow in [-55, -65, -75, -85]:
        plt.plot(bc_curve[pow][0], bc_curve[pow][1], "-", label=f"{pow}dBm", linewidth=2, alpha=0.8)

    plt.xlabel('percentage non common APs')
    plt.ylabel('dis-similarity')
    plt.grid(True)
    plt.rc('font', size=13)
    plt.title(f"disimilarity with non-common APs and {l} common APs")
    # plt.xticks(range(0, 20, 2))
    # plt.xlim(0, 0.55)
    # plt.yticks(np.arange(0, 1, 0.1))
    plt.legend(loc="upper left", shadow=True, fancybox=True)
    plt.show()
    fig.savefig(f"braycurtis-noncommon-APs-near.pdf", bbox_inches='tight')