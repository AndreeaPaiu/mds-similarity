import csv
import random

import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from scipy.linalg import orthogonal_procrustes
from scipy.spatial import procrustes
from scipy.spatial.distance import braycurtis, cosine, correlation, yule
from scipy.stats._mstats_basic import pearsonr
from sklearn.cluster import KMeans
from sklearn.manifold import MDS
import matplotlib.patches as mpatches
import preprocessing_file


from compare_locations import compare_locations
from compute_mds_wifi_similarity import *
from compute_mds_cartesian import *
from pearsonr_similarity import *
from helpers import *

# compute label floor
# compute 2D each floor
# ajust floors with center in 0
# display 3D

colors = []
for color in matplotlib.colors.TABLEAU_COLORS:
    colors.append(color)


def plot_mds_z(collections, simil_method=cosine, n_dim=2, xlabel='Dimensiunea1', ylabel='Dimensiunea2',
               zlabel='Dimnesiunea3', title='', file_name='images/plot.svg', selection='All', add_label=False,
               check_one=False, plot_slope=False, print_angle=False, type_data='wifi', n_clusters=2):
    # pentru a fi sigura ca punctele pot fi random
    # am nevoie de primul punct pentru ordonarea etajelor
    first_point = collections[0].copy()
#     random.shuffle(collections)
    collections = [first_point] + collections

    # compute 3D similarities
    similarities, count_floors = compute_mds_wifi_similarity(collections, simil_method, selection)

    # separ punctele per etaje etajul
    collections = add_floor_label_to_collection(
        collections,
        similarities,
        n_clusters=n_clusters,
        n_dim=n_dim
    )

    # caut ordinea corecta a etajelor
    order = order_floors_using_all_points(collections, similarities)

    # setez etajul corect
    set_right_floor_id(collections, order)

    collections.pop(0)
    # separ pe etaje
    floors_collections = {}
    for collection in collections:
        if collection['floor_id'] not in floors_collections:
            floors_collections[collection['floor_id']] = []
        floors_collections[collection['floor_id']].append(collection)

    _, s = compute_rotation_and_scale(floors_collections[0], simil_method, selection)

    # asamblez atajele
    ref_coordinates_2D = None
    ref_collection = None

    coordinates = compute_2D_mds(
            floors_collections[0],
            simil_method=simil_method,
            ref_coordinates=ref_coordinates_2D,
            selection=selection,
            type_data=type_data,
            ref_collection=ref_collection,
            first_collection=True,
            scale=s
        )


    ref_coordinates_2D = coordinates
    for i in floors_collections:
        if i == 0:
            continue
        ref_collection_id = mapping_floors_nearst_point_distance([floors_collections[i - 1][0], floors_collections[i - 1][1], floors_collections[i - 1][2], floors_collections[i - 1][3]], floors_collections[i])
        _, s = compute_rotation_and_scale(floors_collections[i], simil_method, selection)
        data_coordinates = compute_2D_mds(
            floors_collections[i],
            simil_method=simil_method,
            ref_coordinates=ref_coordinates_2D,
            selection=selection,
            type_data=type_data,
            ref_collection=ref_collection_id,
            scale=s
        )

        ref_coordinates_2D = data_coordinates

        coordinates = coordinates + data_coordinates
#         preprocessing_file.write_csv_mds_and_real_coord(data_coordinates, floors_collections[i])

#     exit()
    # print(coordinates)
    # exit()
    # ref_coordinates_2D = None
    # ref_collection = None
    # for i in floors_collections:
    #     coordinates_3D, ref_coordinates_2D, mtx = compute_2D_mds(
    #         floors_collections[i],
    #         simil_method=simil_method,
    #         ref_coordinates=ref_coordinates_2D,
    #         selection=selection,
    #         type_data=type_data,
    #         ref_collection=ref_collection
    #     )
    #     coordinates = coordinates + coordinates_3D
    #     ref_collection = floors_collections[i]

    # Vizualizarea rezultatelor
    fig = plt.figure()
    plt.axis('equal')
    # # plt.axis([-1, 1, -1, 1])
    # # ax = fig.gca()
    # # ax.set_autoscale_on(False)
    if n_dim == 3:
        ax = fig.add_subplot(111, projection='3d')

    # if n_dim == 2:
    #     plt.xlabel(xlabel)
    #     plt.ylabel(ylabel)
    #     for i, (x, y) in enumerate(coordinates):
    #         plt.scatter(x, y, color=colors[int(collections[i]['floor'])], label=f'Line {i + 1}')
    #
    #     if add_label:
    #         for i, (x, y) in enumerate(coordinates):
    #             plt.text(x + .03, y + .03, collections[i]['floor'], fontsize=9)
    #
    if n_dim == 3:
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_zlabel(zlabel)

        for i, (x, y, z, label) in enumerate(coordinates):
            ax.scatter(x, y, z, color=colors[int(z)], label=f'Line {z}')

        if add_label:
            for i, (x, y, z, label) in enumerate(coordinates):
                # ax.text(x + .03, y + .03,  z + .03, f" {i % len(floors_collections[collections[i]['floor']])}", fontsize=9)
                ax.text(x + .03, y + .03, z + .03, f'{label}',
                        fontsize=9)

    plt.title(title)
    handles = []
    for i in range(0, n_clusters):
        handles.append(mpatches.Patch(color=colors[i], label=f'etaj{i}'))

    plt.legend(handles=handles)
    plt.grid()
    plt.show()
    fig.savefig(file_name, bbox_inches='tight')
