import sys
from processing_file_data import processing_required_data
from helper import *
from datetime import datetime
from scipy.spatial.distance import cosine
from helper_hist import *


import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.manifold import MDS
from _mdsap import *

if __name__ == '__main__':
    # init data
    absolut_path = sys.argv[1]
    files_names = sys.argv[2:]

    # processing data
    merge_data = {}
    data = []
    for control_number, file_name in enumerate(files_names):
        file_data = processing_required_data(absolut_path, file_name, control_number)
        merge_data.update(file_data)
        data.append(file_data)

#     type_data = 'cartesian'
#     type_plot = 'mds'
#     dimension = 3
#
#     # catesian
#     show_data(
#         merge_data,
#         type_data=type_data,
#         type_plot=type_plot,
#         dimension=dimension,
#         path=f'images/raport3/{type_plot}-{dimension}D-{type_data}-points.png',
#         title=f'{type_plot} {dimension}D {type_data} points',
#         xlabel='x',
#         ylabel='y',
#         zlabel='z',
#         simil_method=cosine,
#         selection='All',
#         nr_clusters = len(files_names)
#     )
#
#     type_data = 'wifi'
#     type_plot = 'mds'
#     dimension = 3
#     #all data
#     show_data(
#         merge_data,
#         type_data=type_data,
#         type_plot=type_plot,
#         dimension=dimension,
#         path=f'images/raport3/{type_plot}-{dimension}D-{type_data}-points.png',
#         title=f'{type_plot} {dimension}D {type_data} points',
#         xlabel='x',
#         ylabel='y',
#         zlabel='z',
#         simil_method=cosine,
#         selection='All',
#         nr_clusters = len(files_names)
#     )
#
    # type_data = 'wifi'
    # type_plot = 'mds'
    # dimension = 3
    # # 2 floors
    # for i in range(len(data)):
    #     if i == (len(data) - 1):
    #         break
    #
    #     data_s = {}
    #     data_s.update(data[i])
    #     data_s.update(data[i+1])
    #     show_data(
    #         data_s,
    #         type_data=type_data,
    #         type_plot=type_plot,
    #         dimension=dimension,
    #         path=f'images/raport3/{type_plot}-{dimension}D-{type_data}-points-floor{i}-floor{i+1}.png',
    #         title=f'{type_plot} {dimension}D {type_data} points floor{i} and floor{i+1}',
    #         xlabel='x',
    #         ylabel='y',
    #         zlabel='z',
    #         simil_method=cosine,
    #         selection='All',
    #         nr_clusters = 2
    #     )
#
#     type_data = 'wifi'
#     type_plot = 'mds'
#     dimension = 3
#     #3 floor
#     for i in range(len(data)):
#         if i == (len(data) - 2):
#             break
#
#         data_s = {}
#         data_s.update(data[i])
#         data_s.update(data[i+1])
#         data_s.update(data[i+2])
#         show_data(
#             data_s,
#             type_data=type_data,
#             type_plot=type_plot,
#             dimension=dimension,
#             path=f'images/raport3/{type_plot}-{dimension}D-{type_data}-points-floor{i}-floor{i+1}-floor{i+2}.png',
#             title=f'{type_plot} {dimension}D {type_data} points floor{i} and floor{i+1} and floor{i+2}',
#             xlabel='x',
#             ylabel='y',
#             zlabel='z',
#             simil_method=cosine,
#             selection='All',
#             nr_clusters = 3
#         )
#
#     type_data = 'wifi'
#     type_plot = 'mds'
#     dimension = 2
#     for i in range(len(data)):
#         show_data(
#             data[i],
#             type_data=type_data,
#             type_plot=type_plot,
#             dimension=dimension,
#             path=f'images/raport3/{type_plot}-{dimension}D-{type_data}-points-floor{i}.png',
#             title=f'{type_plot} {dimension}D {type_data} points floor{i}',
#             xlabel='x',
#             ylabel='y',
#             zlabel='z',
#             simil_method=cosine,
#             selection='All',
#             nr_clusters = 2
#         )
#
#     type_data = 'wifi'
#     type_plot = 'mds'
#     dimension = 2
#     show_data(
#         merge_data,
#         type_data=type_data,
#         type_plot=type_plot,
#         dimension=dimension,
#         path=f'images/raport3/{type_plot}-{dimension}D-{type_data}-points.png',
#         title=f'{type_plot} {dimension}D {type_data} points',
#         xlabel='x',
#         ylabel='y',
#         zlabel='z',
#         simil_method=cosine,
#         selection='All',
#         nr_clusters = len(files_names)
#     )
#
#     type_data = 'wifi'
#     type_plot = 'mds'
#     dimension = 3
#     #2 etaje din etaje2 in 2
#     for i in range(len(data)):
#         if i == (len(data) - 2):
#             break
#
#         data_s = {}
#         data_s.update(data[i])
#         data_s.update(data[i+2])
#         show_data(
#             data_s,
#             type_data=type_data,
#             type_plot=type_plot,
#             dimension=dimension,
#             path=f'images/raport3/{type_plot}-{dimension}D-{type_data}-points-floor{i}-floor{i+2}.png',
#             title=f'{type_plot} {dimension}D {type_data} points floor{i} and floor{i+2}',
#             xlabel='x',
#             ylabel='y',
#             zlabel='z',
#             simil_method=cosine,
#             selection='All',
#             nr_clusters = 2
#         )
#
#     type_data = 'wifi'
#     type_plot = 'mds'
#     dimension = 3
#     #2 etaje din 3 in 3
#     for i in range(len(data)):
#         if i == (len(data) - 3):
#             break
#
#         data_s = {}
#         data_s.update(data[i])
#         data_s.update(data[i+3])
#         show_data(
#             data_s,
#             type_data=type_data,
#             type_plot=type_plot,
#             dimension=dimension,
#             path=f'images/raport3/{type_plot}-{dimension}D-{type_data}-points-floor{i}-floor{i+3}.png',
#             title=f'{type_plot} {dimension}D {type_data} points floor{i} and floor{i+3}',
#             xlabel='x',
#             ylabel='y',
#             zlabel='z',
#             simil_method=cosine,
#             selection='All',
#             nr_clusters = 2
#         )
#
# #        2 etaje din 1 in 1
#     type_data = 'wifi'
#     for i in range(len(data)):
#         if i == (len(data) - 1):
#             break
#
#         show_similarities_histogram(
#             data[i],
#             data[i+1],
#             type_data=type_data,
#             path=f'images/raport3/histogram-similarities-{type_data}-points-floor{i}-floor{i+1}.png',
#             title=f'Similarities {type_data} points floor{i} and floor{i+1}',
#             xlabel='Values',
#             ylabel='Density',
#             simil_method=cosine,
#             selection='All'
#         )

#       2 etaje din 1 in 1
#     type_data = 'cartesian'
#     for i in range(len(data)):
#         if i == (len(data) - 1):
#             break
#
#         show_similarities_histogram(
#             data[i],
#             data[i+1],
#             type_data=type_data,
#             path=f'images/raport3/histogram-similarities-{type_data}-points-floor{i}-floor{i+1}.png',
#             title=f'Similarities {type_data} points floor{i} and floor{i+1}',
#             xlabel='Values',
#             ylabel='Density',
#             simil_method=cosine,
#             selection='All'
#         )

# #     2 etaje din 2 in 2
#     type_data = 'wifi'
#     for i in range(len(data)):
#         if i == (len(data) - 2):
#             break
#
#         show_similarities_histogram(
#             data[i],
#             data[i+2],
#             type_data=type_data,
#             path=f'images/raport3/histogram-similarities-{type_data}-points-floor{i}-floor{i+2}.png',
#             title=f'Similarities {type_data} points floor{i} and floor{i+2}',
#             xlabel='Values',
#             ylabel='Density',
#             simil_method=cosine,
#             selection='All'
#         )

# #   2 etaje din 2 in 2
#     type_data = 'cartesian'
#     for i in range(len(data)):
#         if i == (len(data) - 2):
#             break
#
#         show_similarities_histogram(
#             data[i],
#             data[i+2],
#             type_data=type_data,
#             path=f'images/raport3/histogram-similarities-{type_data}-points-floor{i}-floor{i+2}.png',
#             title=f'Similarities {type_data} points floor{i} and floor{i+2}',
#             xlabel='Values',
#             ylabel='Density',
#             simil_method=cosine,
#             selection='All'
#         )


#     type_data = 'wifi'
#     for i in range(len(data)):
#         show_stress_histogram(
#             data[i],
#             type_data=type_data,
#             path=f'images/raport3/histogram-stress-{type_data}-points-floor{i}',
#             title=f'Stress {type_data} points floor{i}',
#             xlabel='Values',
#             ylabel='Density',
#             simil_method=cosine,
#             selection='All',
#             extension='.png'
#         )


##   similaritati intr-un etaj
#     type_data = 'wifi'
#     for i in range(len(data)):
#         show_similarities_histogram(
#             data[i],
#             None,
#             type_data=type_data,
#             path=f'images/raport3/histogram-similarities-{type_data}-points-floor{i}.png',
#             title=f'Similarities {type_data} points floor{i}',
#             xlabel='Values',
#             ylabel='Density',
#             simil_method=cosine,
#             selection='All'
#         )

# #   similaritati intr-un etaj
#     type_data = 'cartesian'
#     for i in range(len(data)):
#         show_similarities_histogram(
#             data[i],
#             None,
#             type_data=type_data,
#             path=f'images/raport3/histogram-similarities-{type_data}-points-floor{i}.png',
#             title=f'Similarities {type_data} points floor{i}',
#             xlabel='Values',
#             ylabel='Density',
#             simil_method=cosine,
#             selection='All'
#         )

# #   testez comportamentul stresului (schimba setul de date)
#     type_data = 'wifi'
#     for i in range(len(data)):
#         show_stress_histogram(
#             data[i],
#             type_data=type_data,
#             path=f'images/raport3/histogram-stress-{type_data}-points-set{i}-84points',
#             title=f'Stress {type_data} points set{i}',
#             xlabel='Values',
#             ylabel='Density',
#             simil_method=cosine,
#             selection='All',
#             extension='.png'
#         )
#
#     type_data = 'wifi'
#     type_plot = 'mds'
#     dimension = 2
#     for i in range(len(data)):
#         show_data(
#             data[i],
#             type_data=type_data,
#             type_plot=type_plot,
#             dimension=dimension,
#             path=f'images/raport3/{type_plot}-{dimension}D-{type_data}-points-set{i}-84points.png',
#             title=f'{type_plot} {dimension}D {type_data} points set{i} 84points',
#             xlabel='x',
#             ylabel='y',
#             zlabel='z',
#             simil_method=cosine,
#             selection='All',
#             nr_clusters = 2
#         )
#
# #      both datasets
#     data_s = {}
#     data_s.update(data[0])
#     data_s.update(data[1])
#     type_data = 'wifi'
#     show_stress_histogram(
#         data_s,
#         type_data=type_data,
#         path=f'images/raport3/histogram-stress-{type_data}-points-both-set{i}-84points',
#         title=f'Stress {type_data} points set{i}',
#         xlabel='Values',
#         ylabel='Density',
#         simil_method=cosine,
#         selection='All',
#         extension='.png'
#     )
#     type_data = 'wifi'
#     type_plot = 'mds'
#     dimension = 2
#     show_data(
#         data_s,
#         type_data=type_data,
#         type_plot=type_plot,
#         dimension=dimension,
#         path=f'images/raport3/{type_plot}-{dimension}D-{type_data}-points-both-set{i}-84points.png',
#         title=f'{type_plot} {dimension}D {type_data} points set{i} 84points',
#         xlabel='x',
#         ylabel='y',
#         zlabel='z',
#         simil_method=cosine,
#         selection='All',
#         nr_clusters = 2
#         )

# #   half dataset
#     type_data = 'wifi'
#     data_s = {}
#     for i, key in enumerate(data[0]):
#         if i % 2 == 0:
#             data_s.update({key: data[0][key]})
#
#     show_stress_histogram(
#         data_s,
#         type_data=type_data,
#         path=f'images/raport3/histogram-stress-{type_data}-points-half-set{0}-84points',
#         title=f'Stress {type_data} points set{0}',
#         xlabel='Values',
#         ylabel='Density',
#         simil_method=cosine,
#         selection='All',
#         extension='.png'
#     )
#
#     type_data = 'wifi'
#     type_plot = 'mds'
#     dimension = 2
#     show_data(
#         data_s,
#         type_data=type_data,
#         type_plot=type_plot,
#         dimension=dimension,
#         path=f'images/raport3/{type_plot}-{dimension}D-{type_data}-points-half-set{0}-84points.png',
#         title=f'{type_plot} {dimension}D {type_data} points set{0} 84points',
#         xlabel='x',
#         ylabel='y',
#         zlabel='z',
#         simil_method=cosine,
#         selection='All',
#         nr_clusters = 2
#         )

#   quarter dataset
#     type_data = 'wifi'
#     data_s = {}
#     for i, key in enumerate(data[0]):
#         if i % 4 == 0:
#             data_s.update({key: data[0][key]})
#
#     show_stress_histogram(
#         data_s,
#         type_data=type_data,
#         path=f'images/raport3/histogram-stress-{type_data}-points-quarter-set{0}-84points',
#         title=f'Stress {type_data} points set{0}',
#         xlabel='Values',
#         ylabel='Density',
#         simil_method=cosine,
#         selection='All',
#         extension='.png'
#     )
#
#     type_data = 'wifi'
#     type_plot = 'mds'
#     dimension = 2
#     show_data(
#         data_s,
#         type_data=type_data,
#         type_plot=type_plot,
#         dimension=dimension,
#         path=f'images/raport3/{type_plot}-{dimension}D-{type_data}-points-quarter-set{0}-84points.png',
#         title=f'{type_plot} {dimension}D {type_data} points set{0} 84points',
#         xlabel='x',
#         ylabel='y',
#         zlabel='z',
#         simil_method=cosine,
#         selection='All',
#         nr_clusters = 2
#         )
#
#
#     type_data = 'wifi'
#     data_s = {}
#     for i, key in enumerate(data[0]):
#         if i % 6 == 0:
#             data_s.update({key: data[0][key]})
#
#     show_stress_histogram(
#         data_s,
#         type_data=type_data,
#         path=f'images/raport3/histogram-stress-{type_data}-points-1-6-set{0}-84points',
#         title=f'Stress {type_data} points set{0}',
#         xlabel='Values',
#         ylabel='Density',
#         simil_method=cosine,
#         selection='All',
#         extension='.png'
#     )

#     type_data = 'wifi'
#     type_plot = 'mds'
#     dimension = 2
#     show_data(
#         data_s,
#         type_data=type_data,
#         type_plot=type_plot,
#         dimension=dimension,
#         path=f'images/raport3/{type_plot}-{dimension}D-{type_data}-points-1-6-set{0}-84points.png',
#         title=f'{type_plot} {dimension}D {type_data} points set{0} 84points',
#         xlabel='x',
#         ylabel='y',
#         zlabel='z',
#         simil_method=cosine,
#         selection='All',
#         nr_clusters = 2
#         )
#
#     type_data = 'cartesian'
#     type_plot = 'mds'
#     dimension = 3
#
#     # catesian
#     show_data(
#         merge_data,
#         type_data=type_data,
#         type_plot=type_plot,
#         dimension=dimension,
#         path=f'images/raport3/{type_plot}-{dimension}D-{type_data}-points.png',
#         title=f'{type_plot} {dimension}D {type_data} points',
#         xlabel='x',
#         ylabel='y',
#         zlabel='z',
#         simil_method=cosine,
#         selection='All',
#         nr_clusters = len(files_names),
#         mds_type = MDSAP
#     )
#
    # type_data = 'wifi'
    # type_plot = 'mds'
    # dimension = 2
    # for i in range(len(data)):
    #     show_data(
    #         data[i],
    #         type_data=type_data,
    #         type_plot=type_plot,
    #         dimension=dimension,
    #         path=f'images/raport3/{type_plot}-{dimension}D-{type_data}-points-floor{i}-with-init-array.png',
    #         title=f'{type_plot} {dimension}D {type_data} points floor{i} with init array',
    #         xlabel='x',
    #         ylabel='y',
    #         zlabel='z',
    #         simil_method=cosine,
    #         selection='All',
    #         nr_clusters = 2,
    #         mds_type = MDSAP
    #     )
#
#     type_data = 'wifi'
#     type_plot = 'mds'
#     dimension = 3
#     # 2 floors
#
#     for i in range(len(data)-3):
#         if i == (len(data)):
#             break
#
#         data_s = {}
#         data_s.update(data[i])
#         data_s.update(data[i+1])
#         data_s.update(data[i+2])
#         print(data_s)
#         ceva =  sorted(data_s, key=select_second_character)
#         data_ss = {}
#         for iter in ceva:
#             data_ss[iter] = data_s[iter]
#         print(data_ss)
#         show_data(
#             data_s,
#             type_data=type_data,
#             type_plot=type_plot,
#             dimension=dimension,
#             path=f'images/raport3/{type_plot}-{dimension}D-{type_data}-points-floor{i}-floor{i+1}.png',
#             title=f'{type_plot} {dimension}D {type_data} points floor{i} and floor{i+1}',
#             xlabel='x',
#             ylabel='y',
#             zlabel='z',
#             simil_method=cosine,
#             selection='All',
#             nr_clusters = 2,
#             mds_type = MDSAP,
#             with_init_array=True
#         )

    type_data = 'wifi'
    type_plot = 'lmbfgs'
    dimension = 2
    for i in range(len(data)):
        show_data(
            data[i],
            type_data=type_data,
            type_plot=type_plot,
            dimension=dimension,
            path=f'images/raport3/{type_plot}-{dimension}D-{type_data}-points-floor{i}-with-init-array.png',
            title=f'{type_plot} {dimension}D {type_data} points floor{i} with init array',
            xlabel='x',
            ylabel='y',
            zlabel='z',
            simil_method=cosine,
            selection='All',
            nr_clusters = 2,
            mds_type = type_plot
        )
