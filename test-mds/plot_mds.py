import random

import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial.distance import braycurtis, cosine, correlation, yule
from scipy.stats._mstats_basic import pearsonr
from sklearn.cluster import KMeans
from sklearn.manifold import MDS
import matplotlib.patches as mpatches

from compare_locations import compare_locations
from compute_mds_wifi_similarity import *
from compute_mds_cartesian import *
from pearsonr_similarity import *
from helpers import f_add_noise


def plot_mds(collections, simil_method=braycurtis, n_dim=2, xlabel='Dimensiunea1', ylabel='Dimensiunea2',
             zlabel='Dimnesiunea3', title='', file_name='images/plot.svg', selection='All', add_label=False, check_one=False, plot_slope=False, print_angle=False, type_data='wifi', n_clusters=1, add_noise=False, range_value=0.5):
    colors = []
    for color in matplotlib.colors.TABLEAU_COLORS:
        colors.append(color)

    # add noise
    if add_noise:
        f_add_noise(collections, range_value)

    similarities = []

    # pentru a fi sigura ca punctele pot fi random
    # random.shuffle(collections)
    # random.shuffle(collections)
    # random.shuffle(collections)
    # random.shuffle(collections)

    # Matricea de similarități între produse
    if type_data == 'wifi':
        similarities, count_floors = compute_mds_wifi_similarity(collections, simil_method, selection)


    if type_data == 'cartesian':
        similarities, count_floors = compute_mds_cartesian(collections, n_dim)

    if check_one:
        count_one = 0
        for i in similarities:
            for j in i:
                if j == 1:
                    count_one += 1

        print(f"similaritati de 1: {count_one}")

    # Crearea unui obiect MDS cu n dimensiuni
    # metric = False => valorile de 0 sunt cosidetare valori lipsa
    mds = MDS(n_components=n_dim, dissimilarity='precomputed')

    # Aplicarea MDS pe matricea de similarități

    coordinates = mds.fit_transform(similarities)

    # clasificare puncte in 2 etaje
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(coordinates)

    add_floor_collections(collections, kmeans.labels_)

    # Vizualizarea rezultatelor
    fig = plt.figure()
    plt.axis('equal')
    # plt.axis([-1, 1, -1, 1])
    # ax = fig.gca()
    # ax.set_autoscale_on(False)
    if n_dim == 3:
        ax = fig.add_subplot(111, projection='3d')

    if plot_slope:
        if n_dim != 3:
            print("panta si unghiurile se afiseaza doar pentru 3D ")
        else:
            x = {}
            y = {}
            z = {}
            floors = []
            # Aplicarea MDS pe matricea de similarități
            coordinates = mds.fit_transform(similarities)
            for r in range(len(coordinates)):
                if int(collections[r]['floor']) not in x:
                    floors.append(int(collections[r]['floor']))
                    x[int(collections[r]['floor'])] = []
                if int(collections[r]['floor']) not in y:
                    y[int(collections[r]['floor'])] = []
                if int(collections[r]['floor']) not in z:
                    z[int(collections[r]['floor'])] = []

                x[int(collections[r]['floor'])].append(coordinates[r][0])
                y[int(collections[r]['floor'])].append(coordinates[r][1])
                z[int(collections[r]['floor'])].append(coordinates[r][2])

            lines = {}
            # Perform linear regression to fit a line to the points
            for i in floors:
                A = []
                A = np.column_stack((x[i], y[i], np.ones_like(x[i])))
                result = np.linalg.lstsq(A, z[i], rcond=None)
                lines[i] = result[0]
                print(f"Slope = {lines[i][:2]}, Intercept = {lines[i][2]}")
                # Generate points for the line of best fit
                x_line, y_line, z_line = [], [], []
                x_line = np.linspace(min(x[i]), max(x[i]), 100)
                y_line = np.linspace(min(y[i]), max(y[i]), 100)
                x_line, y_line = np.meshgrid(x_line, y_line)
                z_line = lines[i][0] * x_line + lines[i][1] * y_line + lines[i][2]
                # Plot the line of best fit in 3D
                ax.plot_surface(x_line, y_line, z_line, alpha=0.2)

            if print_angle:
                for i in range(len(lines) - 1):
                    # Vectorii direcționali pentru fiecare dreaptă
                    v1 = np.array([lines[i][0], lines[i][1], 1])  # Vectorul direcțional pentru prima dreaptă
                    v2 = np.array([lines[i + 1][0], lines[i + 1][1], 1])  # Vectorul direcțional pentru a doua dreaptă

                    # Calculul produsului scalar între vectorii direcționali
                    dot_product = np.dot(v1, v2)

                    # Calculul normelor vectorilor
                    norm_v1 = np.linalg.norm(v1)
                    norm_v2 = np.linalg.norm(v2)

                    # Calculul cosinusului unghiului între vectorii direcționali
                    cosine_angle = dot_product / (norm_v1 * norm_v2)

                    # Calculul unghiului în grade
                    angle = np.arccos(cosine_angle) * 180 / np.pi

                    print(f'Unghiul între cele două dreptele {i} si {i + 1} este: {angle} grade')

    if n_dim == 2:
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        for i, (x, y) in enumerate(coordinates):
            plt.scatter(x, y, color=colors[int(collections[i]['floor_id'])], label=f'Line {i + 1}')

        if add_label:
            for i, (x, y) in enumerate(coordinates):
                plt.text(x + .03, y + .03, collections[i]['label_id'], fontsize=9)



    if n_dim == 3:
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_zlabel(zlabel)

        for i, (x, y, z) in enumerate(coordinates):
            ax.scatter(x, y, z, color=colors[int(collections[i]['floor'])], label=f'Line {i + 1}')

        if add_label:
            for i, (x, y, z) in enumerate(coordinates):
                ax.text(x + .03, y + .03, z + .03, collections[i]['label_id'], fontsize=9)

    plt.title(title)

    handles = []
    for i in range(int(int(collections[len(collections) - 1]['floor_id']) + 1)):
        handles.append(mpatches.Patch(color=colors[i],
                                      label='etaj' + str(i)))
    plt.legend(handles=handles)

    plt.grid()
    plt.show()
    fig.savefig(file_name, bbox_inches='tight')

def plot_all_mds(preprocessing_files_data, simil_methods, selections):
    for simil_method in simil_methods:

        for selection in selections:
            plot_mds(
                preprocessing_files_data[0],
                simil_method=simil_method,
                n_dim=2,
                xlabel='Dimensiunea1',
                ylabel='Dimensiunea2',
                zlabel='Dimnesiunea3',
                title=f'[{simil_method.__name__}] Reprezentare a parterului utilizand MDS',
                file_name=f'images/mds_2D_{simil_method.__name__}_2_floors_with_{selection}_aps_2.png',
                selection=selection,  # Comm | All
                add_label=True,
                plot_slope=False,
                print_angle=False,
                check_one=True,
                type_data='wifi'
            )

def add_floor_collections(collections, labels):
    for i in range(len(collections)):
        collections[i]['floor'] = labels[i]