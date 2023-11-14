import sys

from andreea_test.plot_mds import plot_mds
from plot_similarity_to_nearby_point import plot_similarity_to_nearby_point
from compute_max_dissimilarities_between_signatures_of_the_two_phones_at_the_same_points import \
    compute_max_dissimilarities_between_signatures_of_the_two_phones_at_the_same_points
from plot_similarity_between_points import plot_similarity_between_points
from compute_similarities_using_neighbors import compute_similarities_using_neighbors
from plot_dis_similarity_with_percentage_non_common_aps import *
from andreea_test.compare_locations import *
from preprocessing_file import *
from plot_real_cartesian_system import *
from plot_similarity_using_a_pivot import *

absolut_path = ""

if __name__ == '__main__':
    # set global variables
    absolut_path = sys.argv[1]
    file_name = sys.argv[2]

    preprocessing_file_data = preprocessing_required_data(absolut_path, file_name)

    # print(compare_locations(preprocessing_file_data[0], preprocessing_file_data[1]))
    # plot_real_cartesian_system(preprocessing_file_data)
    # plot_dis_similarity_with_percentage_non_common_aps(preprocessing_file_data[0], preprocessing_file_data[1])
    # plot_similarity_using_a_pivot(preprocessing_file_data)

    # bd = compute_similarities_using_neighbors(preprocessing_file_data)
    # plot_similarity_between_points([[r[2], r[3]] for r in bd])

    file_name_2 = sys.argv[3]
    # preprocessing_file_data_1 = preprocessing_file_data
    preprocessing_file_data_2 = preprocessing_required_data(absolut_path, file_name_2)
    # compute_max_dissimilarities_between_signatures_of_the_two_phones_at_the_same_points(preprocessing_file_data_1, preprocessing_file_data_2)
    # plot_similarity_to_nearby_point(preprocessing_file_data_1, preprocessing_file_data_2)

    plot_mds(preprocessing_file_data_2)
