import numpy as np

from compare_locations import compare_locations


def compute_max_dissimilarities_between_signatures_of_the_two_phones_at_the_same_points(phone_data_1, phone_data_2):
    cl_sim = []
    max_dissim = []
    for nbr in [0, 1, 2, 3]:
        cl_sim.append([])
        max_dissim.append(0)
        for i in range(0, len(phone_data_1)):
            prev = (i - nbr + len(phone_data_1)) % len(phone_data_1)
            next = (i + nbr) % len(phone_data_1)
            if 'wifi' not in phone_data_1[i]:
                continue
            if next >= len(phone_data_2) or next < 0:
                continue
            if prev >= len(phone_data_2) or prev < 0:
                continue

            if 'wifi' not in phone_data_2[next]:
                continue
            # print(prev)
            if 'wifi' not in phone_data_2[prev]:
                continue
            dpr = compare_locations(phone_data_1[i], phone_data_2[next])
            drp = compare_locations(phone_data_1[i], phone_data_2[prev])
            max_dissim[nbr] = max(max_dissim[nbr], dpr, drp)
            cl_sim[nbr].append(dpr)
            cl_sim[nbr].append(drp)
        print(f"Maximum dissimilarity({nbr}) = {max_dissim[nbr]}")
        print(f"CDF[{nbr}] 95/99/99.5 = {np.percentile(cl_sim[nbr], [95, 99, 99.5])}")