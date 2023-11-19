import sys

from plot_mds import plot_mds
from plot_similarity_to_nearby_point import plot_similarity_to_nearby_point
from compute_max_dissimilarities_between_signatures_of_the_two_phones_at_the_same_points import \
    compute_max_dissimilarities_between_signatures_of_the_two_phones_at_the_same_points
from plot_similarity_between_points import plot_similarity_between_points
from compute_similarities_using_neighbors import compute_similarities_using_neighbors
from plot_dis_similarity_with_percentage_non_common_aps import *
from compare_locations import *
from preprocessing_file import *
from plot_real_cartesian_system import *
from plot_similarity_using_a_pivot import *

absolut_path = ""

if __name__ == '__main__':

    absolut_path = sys.argv[1]
    files_names = sys.argv[2:]

    # get all files
    preprocessing_files_data = []
    merged_data = []
    for file_name in files_names:
        file_data = preprocessing_required_data(absolut_path, file_name)
        preprocessing_files_data.append(file_data)
        merged_data += file_data


    # plot_real_cartesian_system(preprocessing_files_data[0])
    # plot_dis_similarity_with_percentage_non_common_aps(preprocessing_files_data[0][0], preprocessing_files_data[0][1])
    # plot_similarity_using_a_pivot(merged_data)

    # bd = compute_similarities_using_neighbors(preprocessing_files_data[0] + preprocessing_files_data[1])
    # plot_similarity_between_points([[r[2], r[3]] for r in bd])

    # compute_max_dissimilarities_between_signatures_of_the_two_phones_at_the_same_points(preprocessing_file_data_1, preprocessing_file_data_2)
    # plot_similarity_to_nearby_point(preprocessing_file_data_1, preprocessing_file_data_2)
    plot_mds(preprocessing_files_data[0] + preprocessing_files_data[1])
