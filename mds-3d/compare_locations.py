import math

import numpy as np
from scipy.spatial.distance import braycurtis


def compare_locations(wifi1, wifi2, simil_method = braycurtis,  selection='All', dif=True, return_type='dis'):
    rssi1 = []
    rssi2 = []

    if selection == 'Comm':
        common_aps = list(set(wifi1.keys()) & set(wifi2.keys()))

        # # No APs in common -> similarity = 1
        if not common_aps:
            return 1.0

        if len(common_aps) < min(len(wifi1.keys()), len(wifi2.keys())) / 4:
            return 1.0

        for ap in common_aps:
            rssi1.append(wifi1[ap])
            rssi2.append(wifi2[ap])

    if selection == 'All':
        all_aps = list(set(set(wifi1.keys()) | set(wifi2.keys())))

        for ap in all_aps:
            if ap in wifi1:
                rssi1.append(wifi1[ap])
            else:
                rssi1.append(0)

            if ap in wifi2:
                rssi2.append(wifi2[ap])
            else:
                rssi2.append(0)

    if return_type == 'rssi':
        return rssi1, rssi2

    return simil_method(rssi1, rssi2)