import numpy as np
from matplotlib import pyplot as plt

from smacofAnchors.MDS_A.test import images
from smacofAnchors.MDS_A.test.config import ALGORITHMS, LABELS, LABELS_MISSING
from smacofAnchors.MDS_A.test.images import plot_3d_real_vs_estimated
from smacofAnchors.MDS_A.test.utils import rmse, compute_estimations, first_third_quartile_and_median, \
    compute_triangulation


def run_rmse_vs_noise(config):
    plot_hanldes = []
    plot_labels = []
    for algorithm in ALGORITHMS:
        config.update_settings(algorithm=algorithm)
        x_axis, errors = rmse_vs_noise(config)
        errors = np.array(errors)
        first_q, median, third_q = first_third_quartile_and_median(errors)
        # print("Algorithm: {}".format(algorithm))
        # print("RMSE: {}".format(rmse(compute_estimations(config), config.points)))
        # print("First Quartile: {}".format(first_q))
        # print("Median: {}".format(median))
        # print("Third Quartile: {}".format(third_q))
        handles = getattr(images, 'plot_{}'.format('rmse_vs_noise'))(first_q, median, third_q, x_axis=x_axis)
        plot_hanldes.extend(handles)
        plot_labels.extend([LABELS[algorithm], 'IQR boundaries'])
    plt.legend(loc='upper left', fontsize=12, labels=plot_labels, handles=plot_hanldes)
    plt.show()
    plt.close('all')

def rmse_vs_noise(config):
    errors = []
    sigmas = np.linspace(0, config.max_sigma, 40)
    for sigma in sigmas:
        config.sigma = sigma
        # generate rmse from coordinates, remembering to not pass last_n_coords to function
        error = [rmse(compute_estimations(config), config.points) for i in range(config.no_of_trials)]
        errors.append(error)
    return sigmas, errors

def run_rmse_vs_anchors(config):
    plot_hanldes = []
    plot_labels = []
    for algorithm in ALGORITHMS:
        config.update_settings(algorithm=algorithm)
        x_axis, errors = rmse_vs_anchors(config)
        errors = np.array(errors)
        first_q, median, third_q = first_third_quartile_and_median(errors)
        # print("Algorithm: {}".format(algorithm))
        # print("RMSE: {}".format(rmse(compute_estimations(config), config.points)))
        # print("First Quartile: {}".format(first_q))
        # print("Median: {}".format(median))
        # print("Third Quartile: {}".format(third_q))
        handles = getattr(images, 'plot_{}'.format('rmse_vs_anchors'))(first_q, median, third_q, x_axis=x_axis)
        plot_hanldes.extend(handles)
        plot_labels.extend([LABELS[algorithm], 'IQR boundaries'])
    plt.legend(loc='upper left', fontsize=12, labels=plot_labels, handles=plot_hanldes)
    plt.show()
    plt.close('all')

def rmse_vs_anchors(config):
    errors = []
    anchors = range(3, 14)
    for anc in anchors:
        config.update_settings(no_of_anchors = anc)
        # generate rmse from coordinates, remembering to not pass last_n_coords to function
        error = [rmse(compute_estimations(config), config.points) for i in range(config.no_of_trials)]
        errors.append(error)
    return anchors, errors


def run_rmse_vs_ntags(config):
    plot_hanldes = []
    plot_labels = []
    algorithm = ALGORITHMS[0]
    # for algorithm in ALGORITHMS:
    config.update_settings(algorithm=algorithm)
    x_axis, errors = rmse_vs_ntags(config)
    errors = np.array(errors)
    first_q, median, third_q = first_third_quartile_and_median(errors)
    # print("Algorithm: {}".format(algorithm))
    # print("RMSE: {}".format(rmse(compute_estimations(config), config.points)))
    # print("First Quartile: {}".format(first_q))
    # print("Median: {}".format(median))
    # print("Third Quartile: {}".format(third_q))
    handles = getattr(images, 'plot_{}'.format('rmse_vs_ntags'))(first_q, median, third_q, x_axis=x_axis)
    plot_hanldes.extend(handles)
    # plot_labels.extend([LABELS[algorithm], 'IQR boundaries'])
    plot_labels.extend([LABELS[algorithm], 'IQR boundaries'])
    plt.legend(loc='upper left', fontsize=12, labels=plot_labels, handles=plot_hanldes)
    plt.show()
    plt.close('all')

def rmse_vs_ntags(config):
    errors = []
    tag_count = np.linspace(10, 1000, 10, dtype=int)
    for nt in tag_count:
        print("Number of tags: {}".format(nt))
        config.update_settings(no_of_tags=nt)
        # generate rmse from coordinates, remembering to not pass last_n_coords to function
        error = [rmse(compute_estimations(config), config.points) for i in range(config.no_of_trials)]
        errors.append(error)
    errors = np.array(errors)
    return tag_count, errors


def run_missing_data_rmse_vs_noise(config):
    plot_hanldes = []
    plot_labels = []
    algorithm = ALGORITHMS[1] # MDS-A
    values = (False, True)
    config.update_settings(algorithm=algorithm)
    for missing_data in values:
        config.update_settings(missingData=missing_data)
        x_axis, errors = rmse_vs_noise(config)
        errors = np.array(errors)
        first_q, median, third_q = first_third_quartile_and_median(errors)
        # print("Algorithm: {}".format(algorithm))
        # print("RMSE: {}".format(rmse(compute_estimations(config), config.points)))
        # print("First Quartile: {}".format(first_q))
        # print("Median: {}".format(median))
        # print("Third Quartile: {}".format(third_q))
        handles = getattr(images, 'plot_{}'.format('rmse_vs_noise'))(first_q, median, third_q, x_axis=x_axis)
        plot_hanldes.extend(handles)
        plot_labels.extend([LABELS_MISSING[missing_data], 'IQR boundaries'])
    plt.legend(loc='upper left', fontsize=12, labels=plot_labels, handles=plot_hanldes)
    plt.show()
    plt.close('all')


def run_rmse_vs_anchor_placement(config):
    plot_handles = []
    plot_labels = []

    # Vom folosi exclusiv MDS-A pentru ambele teste ca să fie o comparație corectă
    config.update_settings(algorithm='_smacof_with_anchors')
    config.update_settings(no_of_anchors=8)  # Folosim 8 ancore

    scenarii = [
        (False, 'Ancore Strategice (Colturi)'),
        (True, 'Ancore Aleatoare (Random)')
    ]

    for is_random, label_name in scenarii:
        config.update_settings(random_anchors=is_random)
        x_axis, errors = rmse_vs_noise(config)
        errors = np.array(errors)
        first_q, median, third_q = first_third_quartile_and_median(errors)
        handles = getattr(images, 'plot_{}'.format('rmse_vs_noise'))(first_q, median, third_q, x_axis=x_axis)
        plot_handles.extend(handles)
        plot_labels.extend([label_name, f'IQR {label_name}'])

    plt.legend(loc='upper left', fontsize=12, labels=plot_labels, handles=plot_handles)
    plt.title("Impactul Plasamentului Ancorelor (Strategic vs Random)")
    plt.show()
    plt.close('all')


def run_real_data(config):
    estimated_points = compute_estimations(config)
    plot_3d_real_vs_estimated(config, estimated_points)


def run_rmse_vs_missing_ratio(config):
    plot_handles = []
    plot_labels = []

    # Alegem manual algoritmul dorit.
    # ALGORITHMS[1] corespunde cu '_smacof_with_anchors' (adică MDS-A)
    # Dacă vrei 'Anchored MDS' (cel clasic), folosește ALGORITHMS[0]
    algorithm = ALGORITHMS[0]

    print(algorithm)
    config.update_settings(algorithm=algorithm)

    # Rulăm simularea
    x_axis, errors = rmse_vs_missing_ratio(config)
    errors = np.array(errors)
    first_q, median, third_q = first_third_quartile_and_median(errors)

    # Generăm graficul
    handles = getattr(images, 'plot_rmse_vs_missing_ratio')(first_q, median, third_q, x_axis=x_axis)
    plot_handles.extend(handles)
    plot_labels.extend([LABELS[algorithm], 'IQR boundaries'])

    plt.legend(loc='upper left', fontsize=12, labels=plot_labels, handles=plot_handles)
    plt.title(f"RMSE Error vs Percentage Missing Data ({LABELS[algorithm]})")
    plt.show()
    plt.close('all')


def rmse_vs_missing_ratio(config):
    errors = []
    ratios = np.linspace(0.0, 1.0, 11)

    for ratio in ratios:
        config.update_settings(missing_data_ratio=ratio)
        config.update_settings(missingData=False)

        # --- NOU: Rulăm o singură dată ca să prindem un "cadru" pentru poză ---
        exemplu_coords = compute_estimations(config)
        images.save_real_vs_estimated_plot(config, exemplu_coords, ratio)
        # ----------------------------------------------------------------------

        # Apoi lăsăm simulările să ruleze cum făceau și până acum (pt linia de pe grafic)
        error = [rmse(compute_estimations(config), config.points) for _ in range(config.no_of_trials)]
        errors.append(error)

    return ratios * 100, errors

# def run_rmse_triangulation(config):
#     errors = []
#     sigmas = np.linspace(0, config.max_sigma, 40)
#     for sigma in sigmas:
#         config.sigma = sigma
#         # generate rmse from coordinates, remembering to not pass last_n_coords to function
#         error = [rmse(compute_triangulation(config), config.points) for i in range(config.no_of_trials)]
#         errors.append(error)
#     return sigmas, errors

# def run_rmse_triangulation(config):
#     plot_handles = []
#     plot_labels = []
#     errors = []
#
#     sigmas = np.linspace(0, config.max_sigma, 40)
#
#     for sigma in sigmas:
#         config.update_settings(sigma=sigma)
#
#         trial_errors = []
#         for i in range(config.no_of_trials):
#             # 1. Apelăm funcția ta originală (asigură-te că ai șters exit() din ea!)
#             coords_tags = compute_triangulation(config)
#
#             # 2. Reconstruim formatul complet (Ancore + Tag-uri) ca să se potrivească cu config.points
#             puncte_estimate = list(config.anchors.copy())
#             for estimare in coords_tags:
#                 if estimare is not None:
#                     # Dacă rețeaua e 3D, dar formula dă doar X, Y, punem 0 pe axa Z
#                     if config.n_components == 3 and len(estimare) == 2:
#                         estimare = np.append(estimare, 0.0)
#                     puncte_estimate.append(estimare)
#                 else:
#                     # Dacă inelele ratează intersecția, punem NaN ca să știm că e invalid
#                     puncte_estimate.append(np.full(config.n_components, np.nan))
#
#             puncte_estimate = np.array(puncte_estimate)
#
#             # 3. Filtrăm array-urile ca să trimitem către funcția ta rmse DOAR punctele care s-au putut calcula
#             masca_valide = ~np.isnan(puncte_estimate).any(axis=1)
#             valid_computed = puncte_estimate[masca_valide]
#             valid_real = config.points[masca_valide]
#
#             # 4. Apelăm funcția ta ORIGINALĂ rmse (nefiind modificate, ambele array-uri au acum aceeași formă curată)
#             if len(valid_computed) > 0:
#                 eroare = rmse(valid_computed, valid_real)
#             else:
#                 eroare = np.nan  # Totul a picat la acest zgomot
#
#             trial_errors.append(eroare)
#
#         errors.append(trial_errors)
#
#     errors = np.array(errors)
#
#     # 5. Calculăm medianele și cuartilele ignorând eventualele trial-uri picate (NaN)
#     first_q = np.nanpercentile(errors, 25, axis=1)
#     third_q = np.nanpercentile(errors, 75, axis=1)
#     median = np.nanmedian(errors, axis=1)
#
#     # 6. Desenăm graficul folosind sistemul tău standard din images.py
#     handles = getattr(images, 'plot_rmse_vs_noise')(first_q, median, third_q, x_axis=sigmas)
#     plot_handles.extend(handles)
#     plot_labels.extend(['Trilateration', 'IQR boundaries'])
#
#     plt.legend(loc='upper left', fontsize=12, labels=plot_labels, handles=plot_handles)
#     plt.title("RMSE Error vs Measurement Noise (Trilateration)")
#     plt.show()
