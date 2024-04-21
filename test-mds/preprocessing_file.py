import json
import math
import sys

import numpy as np
import pandas as pd


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


def get_real_cartesian_coordinates(data_file):
    real_cartesian_coordinates = {}

    for collection_key in data_file:
        collection = data_file[collection_key]
        if 'x' not in collection or 'y' not in collection:
            continue

        if collection_key not in real_cartesian_coordinates:
            real_cartesian_coordinates[collection_key] = {}

        real_cartesian_coordinates[collection_key] = [collection['x'], collection['y'], collection['z']]

        if len(real_cartesian_coordinates[collection_key]) == 0:
            print(collection_key)
            real_cartesian_coordinates.pop(collection_key)
        collection.pop('x')
        collection.pop('y')
        collection.pop('z')
    return real_cartesian_coordinates


def get_wifi_rssi(absolut_path, data_file):
    result = {}
    for collection_key in data_file:
        collection = data_file[collection_key]
        if 'fingerprints' not in collection:
            continue
        result[collection_key] = {}
        fingerprints = collection['fingerprints']
        for fingerprint in fingerprints:
            if 'wifi' not in fingerprint:
                continue

            for mac in fingerprint['wifi']:
                avg_pow = np.average(fingerprint["wifi"][mac]['rssi'])

                if mac not in result[collection_key]:
                    result[collection_key][mac] = {'rssi': 0}

                result[collection_key][mac]['rssi'] = adjust_rssi(avg_pow)[0]
                # result[collection_key][mac]['rssi'] = avg_pow

                if result[collection_key][mac]['rssi'] == 0:
                    result.pop(collection_key)

    return result


def preprocessing_required_data(absolut_path, file_name, floor):
    # get file data
    file_data = read_json_file(absolut_path, file_name)

    result = {}

    # get real cartesian coordinates
    real_cartesian_coordinates = get_real_cartesian_coordinates(file_data)

    # get wifi rssi
    wifi_rssi = get_wifi_rssi(absolut_path, file_data)

    for collection_key in file_data:
        if collection_key not in result:
            result[collection_key] = {}

        if collection_key in real_cartesian_coordinates:
            result[collection_key]['real_coordinates'] = real_cartesian_coordinates[collection_key]

        if collection_key in wifi_rssi:
            result[collection_key]['wifi'] = wifi_rssi[collection_key]

    result_array = []
    i = 0
    for key in result:
        if result[key] != {}:
            result[key]['label_id'] = i
            result[key]['floor_id'] = floor
            result_array.append(result[key])
            i += 1

    return result_array

def write_csv_mds_and_real_coord(coord_mds, coord_real):
    for i in range(len(coord_mds)):
        coord_mds[i] = coord_mds[i] + coord_real[i]['real_coordinates']
    # np.savetxt(f"coords_mds_real_data/data{coord_real[0]['floor_id']}.csv", coord_mds,
    #            delimiter=" ")

    DF = pd.DataFrame(coord_mds)

    # save the dataframe as a csv file
    DF.to_csv(f"coords_mds_real_data/data{coord_real[0]['floor_id']}.csv")
