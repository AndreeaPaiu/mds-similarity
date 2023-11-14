import numpy as np
from scipy.spatial.distance import braycurtis


def compare_locations(c1, c2, simil_method = braycurtis,  selection = 'Average', dif = True, return_type='dis'):
    wifi1 = c1['wifi']
    wifi2 = c2['wifi']

    common_aps = list(set(wifi1.keys()) & set(wifi2.keys()))

    # No APs in common -> similarity = 1
    if not common_aps:
        return 1.0

    if len(common_aps) * 3 <= len(wifi1.keys()) or len(common_aps) < 3:
        return 1.0

    # numarul de ap diferite
    aps1 = set(wifi1.keys()) - set(common_aps)
    aps2 = set(wifi2.keys()) - set(common_aps)

    rssi1 = []
    rssi2 = []

    if selection == 'Average':
        for ap in common_aps:
            rssi1.append(wifi1[ap]['rssi'])
            rssi2.append(wifi2[ap]['rssi'])

    if return_type == 'rssi':
        return rssi1, rssi2

    return braycurtis(rssi1, rssi2)

