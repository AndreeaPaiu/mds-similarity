import sys

import numpy as np
from scipy.spatial.distance import cosine

from smacofAnchors.MDS_A.test.config import SettingsManager
from smacofAnchors.MDS_A.test.tests import run_rmse_vs_noise, run_rmse_vs_anchors, run_rmse_vs_ntags, \
    run_missing_data_rmse_vs_noise, run_rmse_vs_anchor_placement, run_real_data
from smacofAnchors.smacofAnchors2026 import plot_wifi_vs_physical_distance, extrage_profil_eroare
from test_mds.compute_mds_wifi_similarity import compute_mds_wifi_similarity
from test_mds.preprocessing_file import preprocessing_required_data
from smacofAnchors.MDS_A.test.tests import run_rmse_vs_missing_ratio

if __name__ == '__main__':
    config = SettingsManager()
    config.set_random_tags()

    # variaza zgomotul
    # test 1
    # run_rmse_vs_noise(config)

    # test 2
    # config.update_settings(missingData = True)
    # run_rmse_vs_noise(config)

    # variaza numarul de ancore
    # test 3
    # config.update_settings(sigma = 0)
    # run_rmse_vs_anchors(config)

    # test 4
    # config.update_settings(missingData = True)
    # config.update_settings(sigma=0)
    # run_rmse_vs_anchors(config)

    # test 5
    # config.update_settings(sigma = 1)
    # run_rmse_vs_anchors(config)

    # test 6
    # config.update_settings(sigma = 2)
    # run_rmse_vs_anchors(config)

    # test 5
    # config.update_settings(sigma = 1)
    # config.update_settings(missingData=True)
    # run_rmse_vs_anchors(config)

    # test 6
    # config.update_settings(sigma = 2)
    # config.update_settings(missingData = True)
    # run_rmse_vs_anchors(config)

    # test 7
    # config.update_settings(sigma=4)
    # config.update_settings(missingData=True)
    # run_rmse_vs_anchors(config)

    # variez numarul de tags
    # test 8
    # config.update_settings(sigma=4)
    # config.update_settings(no_of_tags=100)
    # run_rmse_vs_ntags(config)

    # test 9
    # config.update_settings(missingData=True)
    # config.update_settings(sigma=4)
    # run_rmse_vs_ntags(config)

    # schimb sigma de la 0 la 10 in 2D
    # test 10
    # config.update_settings(max_sigma = 10)
    # run_missing_data_rmse_vs_noise(config)

    # test 11 -> schimb sigma de la 0 la 10 in 3D
    # config.update_settings(n_components=3)
    # config.update_settings(max_sigma=10)
    # run_missing_data_rmse_vs_noise(config)

    # test 12 -> schimb sigma de la 0 la 10 in 3D
    # config.update_settings(n_jobs=10)
    # config.update_settings(n_components=3)
    # config.update_settings(max_sigma=10)
    # run_missing_data_rmse_vs_noise(config)

    # compar o pozitionare centrala si una random
    # test 13
    # config.update_settings(max_sigma=10)
    # run_rmse_vs_anchor_placement(config)

    # testat pe date reale
    absolut_path = sys.argv[1]
    files_names = sys.argv[2:]
    #
    # # get all files
    # preprocessing_files_data = []
    # merged_data = []
    #
    # for file_name in files_names[:2]:
    #     file_data = preprocessing_required_data(absolut_path, file_name, file_name[0])
    #     preprocessing_files_data.append(file_data)
    #     merged_data += file_data
    #
    # print(len(merged_data))
    #
    # # similarities, _ = compute_mds_wifi_similarity(merged_data, cosine, 'All')
    # config.update_settings(real_data=np.array(merged_data))
    # config.update_settings(is_real_data=True)
    # config.update_settings(no_of_anchors=4)
    # config.update_settings(no_of_tags=len(merged_data) - 4)
    # config.update_settings(is_random=False)
    # config.process_real_data()
    #
    # run_real_data(config)
    # # test 14
    # run_real_data_rmse_vs_anchors(config, merged_data, similarities)

    # test 15 -> noise = 4, Variază procentul de date tag-to-tag lipsă
    # config.update_settings(sigma=4)
    # config.update_settings(no_of_tags=30)
    # run_rmse_vs_missing_ratio(config)

    # test 16 -> noise = 4

    # config.update_settings(x_max=500)
    # config.update_settings(y_max=500)
    # config.update_settings(z_max=500)
    # config.update_settings(sigma=4 )
    # # # config.update_settings(no_of_tags=500)
    # config.update_settings(no_of_tags=100)
    # run_rmse_vs_missing_ratio(config)

    # test 17
    # config.update_settings(no_of_anchors=4)
    # run_rmse_triangulation(config)


    # test 18
    # config.update_settings(no_of_anchors=3)
    # config.update_settings(use_triangulation_init=True)
    # config.update_settings(missingData=True)
    # run_rmse_vs_noise(config)

    # test 19
    # config.update_settings(sigma=4)
    # config.update_settings(no_of_tags=100)
    # config.update_settings(n_init=1)
    # config.update_settings(max_iter=50)
    # config.update_settings(missingData=True)
    # config.update_settings(use_triangulation_init=True)
    # run_rmse_vs_ntags(config)

    # config.update_settings(x_max=500)
    # config.update_settings(y_max=500)
    # config.update_settings(z_max=500)
    # config.update_settings(sigma=4 )
    # config.update_settings(n_init=1)
    # config.update_settings(max_iter=50)
    # config.update_settings(no_of_tags=500)
    # # config.update_settings(no_of_tags=100)
    # config.update_settings(use_triangulation_init=True)
    # run_rmse_vs_missing_ratio(config)



    ########################
    # get all files
    preprocessing_files_data = []
    merged_data = []

    for file_name in files_names[:2]:
        file_data = preprocessing_required_data(absolut_path, file_name, file_name[0])
        preprocessing_files_data.append(file_data)
        merged_data += file_data

    print(len(merged_data))

    similarities, _ = compute_mds_wifi_similarity(merged_data, cosine, 'All')
    date_wifi = merged_data
    matrice_D_reala = similarities

    # plot_wifi_vs_physical_distance(date_wifi, matrice_D_reala, max_dist=7.0)

    mu_real, sigma_reala = extrage_profil_eroare(date_wifi, matrice_D_reala, max_dist=7.0)

    # 3. Injectăm profilul de eroare real înapoi în configurarea testelor tale!
    config.update_settings(mu=mu_real)
    config.update_settings(sigma=sigma_reala)

    print(f"Am updatat config: MDS/Trilateratia va rula cu zgomotul cladirii tale (sigma={config.sigma:.2f}m)!")
    # config.update_settings( real_data=np.array(merged_data))
    # config.update_settings(is_real_data=True)
    # config.update_settings(no_of_anchors=4)
    # config.update_settings(no_of_tags=len(merged_data) - 4)
    # config.update_settings(is_random=False)
    # config.process_real_data()

    # run_real_data(config)
    # test 14
    # run_real_data_rmse_vs_anchors(config, merged_data, similarities)