import sys
from preprocessing_file import *
import json

if __name__ == '__main__':
    absolut_path = sys.argv[1]
    files_names = sys.argv[2:]

    for file_name in files_names:
        file_data = read_json_file(absolut_path, file_name)
        keys = list(file_data.keys())
        data = {}

        for i in range(len(keys) - 1):
            if i % 2 == 0:
                file_data[keys[i]]['fingerprints'].append(file_data[keys[i + 1]]['fingerprints'][0])
                data[keys[i]] = file_data[keys[i]]

        with open("C:\\Users\\paiua\\Desktop\\work_project\\cercetare\\mds-similarity\\test-mds\\data_andreea_pereche\\test" + file_name,
                  'w') as json_file:
            json.dump(data, json_file, indent=4)
