import sys

from numpy.linalg import norm
from scipy.stats._mstats_basic import pearsonr
from sklearn.metrics.pairwise import cosine_similarity

from plot_mds import plot_mds, plot_all_mds
from plot_similarity_to_nearby_point import plot_similarity_to_nearby_point
from compute_max_dissimilarities_between_signatures_of_the_two_phones_at_the_same_points import \
    compute_max_dissimilarities_between_signatures_of_the_two_phones_at_the_same_points
from plot_similarity_between_points import plot_similarity_between_points, plot_all_similarity_between_points
from compute_similarities_using_neighbors import compute_similarities_using_neighbors
from plot_dis_similarity_with_percentage_non_common_aps import *
from compare_locations import *
from preprocessing_file import *
from plot_real_cartesian_system import *
from plot_similarity_using_a_pivot import *
from k_means_mds import *
from plot_mds_real_cartesian_system import *
from plot_mds_braycurtis import *
from plot_mds_cosine import *
from plot_mds_pearsonr import *
from plot_mds_manhattan import *
from plot_mds_3D_real_cartesian_system import *
from plot_mds_3D_braycurtis import *
from plot_mds_3D_cosine import *
from plot_mds_3D_pearsonr import *
from plot_mds_3D_manhattan import *
from pearsonr_similarity import *
from plot_mds_z import *

absolut_path = ""
simil_methods = [braycurtis, cosine, pearsonr_similarity, correlation, yule]
selections = ['All']

if __name__ == '__main__':

    absolut_path = sys.argv[1]
    files_names = sys.argv[2:]

    # get all files
    preprocessing_files_data = []
    merged_data = []

    for file_name in files_names:
        file_data = preprocessing_required_data(absolut_path, file_name, file_name[0])
        preprocessing_files_data.append(file_data)
        merged_data += file_data

    plot_mds(
        preprocessing_files_data[0] + preprocessing_files_data[1] + preprocessing_files_data[2] + preprocessing_files_data[3],
        simil_method=cosine,
        n_dim=3,
        xlabel='Dimensiunea1',
        ylabel='Dimensiunea2',
        zlabel='Dimnesiunea3',
        title='[cosine][All Aps] Reprezentarea a 2 etaje utilizand MDS',
        file_name='images/label_mds_3D_cosine_2_floors',
        selection='All',  # Comm
        add_label=True,
        plot_slope=True,
        print_angle=True,
        check_one=True,
        type_data='wifi',
        n_clusters=2
    )

    plot_mds_z(
        merged_data,
        simil_method=cosine,
        n_dim=3,
        xlabel='Dimensiunea1',
        ylabel='Dimensiunea2',
        zlabel='Dimnesiunea3',
        title='[cosine][All Aps] Reprezentarea a 2 etaje utilizand MDS',
        file_name='images/label_mds_3D_cosine_2_floors',
        selection='All',  # Comm
        add_label=True,
        plot_slope=False,
        print_angle=False,
        check_one=False,
        type_data='wifi',
        n_clusters=8
    )

    # plot_similarity_between_points(
    #     preprocessing_files_data[0] + preprocessing_files_data[1],
    #     selection='Comm',  # Comm | All
    #     simil_method=yule,
    #     title="Dataset 1b yule vs distance",
    #     file_name="yule-vs-distance-ds1.svg"
    # )

    # plot_all_similarity_between_points(
    #     preprocessing_files_data[0] + preprocessing_files_data[1],
    #     simil_methods=simil_methods,
    #     selections=selections
    # )

    # plot_all_mds(preprocessing_files_data, simil_methods, selections)

    # plot_real_cartesian_system(preprocessing_files_data[0] +
    #     preprocessing_files_data[2] +
    #     preprocessing_files_data[4] +
    #     preprocessing_files_data[6] +
    #     preprocessing_files_data[8] +
    #     preprocessing_files_data[10] +
    #     preprocessing_files_data[12] +
    #     preprocessing_files_data[14]
    #                            )

    # plot_real_cartesian_system(preprocessing_files_data[0])

    # plot_mds_real_cartesian_system(preprocessing_files_data[0] + preprocessing_files_data[1])
    # plot_mds_real_cartesian_system(merged_data)
    # plot_mds_3D_real_cartesian_system(preprocessing_files_data[0] + preprocessing_files_data[2])

    # plot_mds_braycurtis(preprocessing_files_data[0] + preprocessing_files_data[1])
    # plot_mds_braycurtis(merged_data)
    # plot_mds_3D_braycurtis(preprocessing_files_data[0] + preprocessing_files_data[1] + preprocessing_files_data[2] + preprocessing_files_data[3])
    # plot_mds_3D_braycurtis(preprocessing_files_data[0] + preprocessing_files_data[2])

    # plot_mds_cosine(preprocessing_files_data[0] + preprocessing_files_data[1])
    # plot_mds_cosine(merged_data)
    # plot_mds_3D_cosine(
    #     preprocessing_files_data[0] +
    #     preprocessing_files_data[2] +
    #     preprocessing_files_data[4] +
    #     preprocessing_files_data[6] +
    #     preprocessing_files_data[8] +
    #     preprocessing_files_data[10] +
    #     preprocessing_files_data[12] +
    #     preprocessing_files_data[14], merged_aps)
    # plot_mds_3D_cosine(preprocessing_files_data[0] + preprocessing_files_data[2])

    # plot_mds_pearsonr(preprocessing_files_data[0] + preprocessing_files_data[1])
    # plot_mds_pearsonr(merged_data)
    # plot_mds_3D_pearsonr(preprocessing_files_data[0] + preprocessing_files_data[2])


    # plot_mds_manhattan(preprocessing_files_data[0] + preprocessing_files_data[1])
    # plot_mds_manhattan(merged_data)
    # plot_mds_3D_manhattan(preprocessing_files_data[0] + preprocessing_files_data[2])

    # plot_dis_similarity_with_percentage_non_common_aps(preprocessing_files_data[0][0], preprocessing_files_data[0][1])
    # plot_similarity_using_a_pivot(merged_data)

    # compute_max_dissimilarities_between_signatures_of_the_two_phones_at_the_same_points(preprocessing_file_data_1, preprocessing_file_data_2)
    # plot_similarity_to_nearby_point(preprocessing_file_data_1, preprocessing_file_data_2)

    # plot_mds(merged_data)
    # k_means_mds(merged_data)

    # print(compare_locations(preprocessing_files_data[0][0], preprocessing_files_data[0][1]))

    # import numpy as np
    from numpy.linalg import norm
    # A = [0.1, 0.2, 0.3, 0]
    # B = [0, 0.2, 0.3, 0]
    # C = [0.1, 0.2, 0.3, 0.4]
    # D = [0.4, 0.2, 0.3, 0]
    # print(braycurtis(A, B))
    # print(braycurtis(C, B))
    # print(braycurtis(D, B))
    # print(cosine(A, B))
    # print(cosine(C, B))
    # print(cosine(D, B))
    # print(yule([1, 0, 0, 1], [0, 0, 1, 1]))
    # print(yule([1, 0, 0, 0], [0, 0, 1, 0]))

    # print(cosine_similarity(np.array([2, 3]),np.array( [2, 3])))
