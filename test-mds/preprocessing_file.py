import json
import sys

import numpy as np


def read_json_file(absolut_path, file_name):
    file = open(absolut_path + file_name)
    data = json.load(file)
    file.close()

    return data


def get_real_cartesian_coordinates(data_file):
    real_cartesian_coordinates = {}

    for collection_key in data_file:
        collection = data_file[collection_key]
        if 'x' not in collection or 'y' not in collection:
            continue
        if collection_key not in real_cartesian_coordinates:
            real_cartesian_coordinates[collection_key] = {}

        real_cartesian_coordinates[collection_key] = [collection['x'], collection['y'], collection['z']]
        collection.pop('x')
        collection.pop('y')
        collection.pop('z')
    return real_cartesian_coordinates


def get_wifi_rssi(absolut_path, data_file):
    wifi_rssi = {}
    # whitelist_data = read_json_file(absolut_path, "whitelist.json")

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
                if 'rssi' not in fingerprint['wifi'][mac]:
                    continue

                avg_pow = np.average(fingerprint["wifi"][mac]['rssi'])

                if mac not in result[collection_key]:
                    result[collection_key][mac] = {'rssi': 0}

                result[collection_key][mac]['rssi'] = avg_pow
    return result

def preprocessing_required_data(absolut_path, file_name):
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
    for key in result:
        result_array.append(result[key])

    return result_array
