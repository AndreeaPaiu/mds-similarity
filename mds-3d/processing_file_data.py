import json
import math
import sys
import numpy as np
import pandas as pd
import re
# from sklearn.manifold import MDS
# from compute_mds_wifi_similarity import *
# from scipy.spatial.distance import braycurtis, cosine, correlation, yule
#
#
#
def read_json_file(absolut_path, file_name):
    file = open(absolut_path + file_name)
    data = json.load(file)
    file.close()

    return data

# powed like in Sospedra Comprehensive, but scaled so that -100dBm=0 -30dBm=1
adjust_rssi_params = -100, math.e, 2.63

# more penalty for low dBm
# adjust_rssi_params = -100, 2.0*math.e, 6.95

# less penalty for high dBm
# adjust_rssi_params = -100, 0.7*math.e, 1.97

# rss_in param is int or list
# rss_out is a list
def adjust_rssi(rssi_in):
    min_rssi, exponent, scaler = adjust_rssi_params
    rss_out = np.array([])
    if (type(rssi_in) is int) or (type(rssi_in) is float) or (type(rssi_in) is np.float64) or (
            type(rssi_in) is np.int64):
        rssi_in = np.array([rssi_in])
    if type(rssi_in) is list:
        rssi_in = np.array(rssi_in)
    for rssi_val in rssi_in:
        if rssi_val < 0 and rssi_val > min_rssi:
            positive = rssi_val - min_rssi
            rssi = scaler * pow(-positive / min_rssi, exponent)
        else:
            rssi = 0
        rss_out = np.append(rss_out, rssi)
    return rss_out

def processing_wifi_fingerprints(fingerprints):

    wifi_rssi = {}
    if not fingerprints:
        return wifi_rssi

    for fingerprint in fingerprints:
        if 'wifi' not in fingerprint:
                continue

        for mac in fingerprint['wifi']:
            if mac not in wifi_rssi:
                wifi_rssi[mac] = []

            if 'rssi' not in fingerprint["wifi"][mac]:
                continue

            wifi_rssi[mac] += fingerprint["wifi"][mac]['rssi']

    for mac in wifi_rssi:
        avg_pow = np.average(wifi_rssi[mac])
        wifi_rssi[mac] = adjust_rssi(avg_pow)[0]

    return wifi_rssi


def processing_data(collections, control_number):

    fingerprints = {}
    for collection_key in collections:
        if not re.search("collection*", collection_key):
            continue

        collection = collections[collection_key]

        if not collection:
            continue

        fingerprints[str(control_number) + collection_key] = {}

        # get cartesian coordinates
        fingerprints[str(control_number) + collection_key]['cartesian_coordinates'] = []
        if 'x' in collection and 'y' in collection and 'z' in collection:
            fingerprints[str(control_number) + collection_key]['cartesian_coordinates'] = [collection['x'], collection['y'], collection['z']]

        # process fingerprints
        fingerprints[str(control_number) + collection_key]['wifi'] = []

        if 'fingerprints' in collection:
            # process wifi fingerprint
            fingerprints[str(control_number) + collection_key]['wifi'] = processing_wifi_fingerprints(collection['fingerprints'])

    return fingerprints

def processing_required_data(absolut_path, file_name, control_number):
    # get file data
    file_data = read_json_file(absolut_path, file_name)

    # process fingerprints
    data = processing_data(file_data, control_number)

    return data
