from datetime import datetime

import numpy as np
from scipy.spatial.distance import euclidean, cosine, braycurtis

from compare_locations import compare_locations


def compute_similarities_using_neighbors(etaj_data):
    before = datetime.now()
    # TP = punct in limite
    # FP = punct cu disimilaritate in limte dar prea departe
    # FN = punct cu disimilaritate mare, dar aproape
    TP, FP, FN = 0, 0, 0
    fpd, posd = [], []
    bd = []
    # R = distanta maxima intre puncte
    # thr = disimilaritatea maxima
    R, thr = 3.0, 0.32

    for q in range(0, len(etaj_data) - 1):
        for p in range(q + 1, len(etaj_data)):
            if 'real_coordinates' in etaj_data[q] and 'real_coordinates' in etaj_data[p]:
                eu_dist = euclidean([etaj_data[q]['real_coordinates'][0], etaj_data[q]['real_coordinates'][1], etaj_data[q]['real_coordinates'][2]],
                                    [etaj_data[p]['real_coordinates'][0], etaj_data[p]['real_coordinates'][1], etaj_data[p]['real_coordinates'][2]])

                bc = compare_locations(etaj_data[q], etaj_data[p], selection='Average', dif=True, simil_method=braycurtis)
                # cor = compare_locations(etaj_data[q], etaj_data[p], selection='Average', dif=True, simil_method=cosine)

                if bc < thr:
                    if eu_dist < R:
                        TP = TP + 1
                    else:
                        FP = FP + 1
                        fpd.append(eu_dist)
                    posd.append(eu_dist)
                else:
                    if eu_dist < R:
                        FN = FN + 1

                bd.append([q, p, eu_dist, bc])
    after = datetime.now()

    print(f"R = {R} bc= {thr} Precision= {TP * 1.0 / (TP + FP):.2f} Recall= {TP * 1.0 / (TP + FN):.2f} Pdist = {np.percentile(posd, [50, 95, 99])}")
    m = np.polyfit([r[2] for r in bd], [r[3] for r in bd], 1)
    print(f"Slope = {m[0]:.2f}")

    return bd