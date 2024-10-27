import numpy as np
import copy
from scipy.linalg import orthogonal_procrustes
from scipy.spatial import procrustes
import matplotlib.pyplot as plt
from numpy import linalg as LA
import math
import seaborn as sns

def get_hist():
    columns = ['idx', 'mds_x', 'mds_y', 'mds_z', 'x', 'y', 'z']
    diff_lines = []
    for i in range(7):
        DATA_0_FILE = f'C:\\Users\\paiua\\Desktop\\work_project\\cercetare\\mds-similarity\\test-mds\\coords_mds_real_data\\data{i}-pixel-1.csv'
        sim_data_0 = np.genfromtxt(DATA_0_FILE, delimiter=',', skip_header=1, names=columns, dtype=None)
        Line1 = copy.deepcopy(sim_data_0)

        matrix1 = []
        realmatrix1 = []

        for i in Line1:
            matrix1.append([i['mds_x'], i['mds_y']])
            realmatrix1.append([i['x'], i['y']])

        matrix1 = np.array(matrix1)
        realmatrix1 = np.array(realmatrix1)

        media = np.mean(realmatrix1, axis = 0)
        realmatrix1 = realmatrix1 - media

        mtx_r1 = np.array([realmatrix1[0], realmatrix1[12], realmatrix1[25], realmatrix1[37]])
        mtx_r2 = np.array([matrix1[0], matrix1[12], matrix1[25], matrix1[37]])
        R, s = orthogonal_procrustes(mtx_r1, mtx_r2)
        print(s)
        print(R.T)
        mtx1 = realmatrix1
        mtx2 = matrix1@R.T

        # std1 = np.std(mtx_r2)
        # std2 = np.std(mtx_r1)

        std1 = np.std(mtx2)
        std2 = np.std(mtx1)

        # Scalăm prima matrice la aceeași scară cu a doua matrice
        mtx2 = mtx2 / std1 * std2

        for i in range(len(mtx1)):
            diff_lines.append(math.dist(mtx1[i], mtx2[i]))

    fig = plt.figure()
    sns.histplot(diff_lines, bins=int(max(diff_lines) - min(diff_lines)), kde=True, color='skyblue', edgecolor='black')
    print(int(max(diff_lines) - min(diff_lines)))
    print(len(diff_lines))
    print(f"max = {max(diff_lines)}")
    print(f"min = {min(diff_lines)}")
    print(f"avg = {np.average(diff_lines)}")

    # Adding labels and title
    plt.xlabel('Distance')
    plt.ylabel('Frequency')
    plt.title('Errors')

    # Display the plot
    plt.show()
    fig.savefig("../test-mds/images/raport-2/procrustes_histogram_all_floor_mds_real.png", bbox_inches='tight')

def get_hist_only_mds():
    columns = ['idx', 'mds_x', 'mds_y']
    diff_lines = []
    for i in range(7):
        DATA_0_FILE = f'C:\\Users\\paiua\\Desktop\\work_project\\cercetare\\mds-similarity\\test-mds\\raport-3\\coords_mds_data\\data-andreea-2.csv'
        DATA_1_FILE = 'C:\\Users\\paiua\\Desktop\\work_project\\cercetare\\mds-similarity\\test-mds\\raport-3\\coords_mds_data\\data-andreea-2.csv'
        DATA_2_FILE = 'C:\\Users\\paiua\\Desktop\\work_project\\cercetare\\mds-similarity\\test-mds\\raport-3\\coords_mds_data\\data-andreea-3.csv'
        DATA_3_FILE = 'C:\\Users\\paiua\\Desktop\\work_project\\cercetare\\mds-similarity\\test-mds\\raport-3\\coords_mds_data\\data-andreea-4.csv'

        sim_data_0 = np.genfromtxt(DATA_0_FILE, delimiter=',', skip_header=1, names=columns, dtype=None)
        sim_data_1 = np.genfromtxt(DATA_1_FILE, delimiter=',', skip_header=1, names=columns, dtype=None)
        sim_data_2 = np.genfromtxt(DATA_2_FILE, delimiter=',', skip_header=1, names=columns, dtype=None)
        sim_data_3 = np.genfromtxt(DATA_3_FILE, delimiter=',', skip_header=1, names=columns, dtype=None)

        Line1 = copy.deepcopy(sim_data_0)
        Line2 = copy.deepcopy(sim_data_1)
        Line3 = copy.deepcopy(sim_data_2)
        Line4 = copy.deepcopy(sim_data_3)

        matrix1 = []
        for i in Line1:
            matrix1.append([i['mds_x'], i['mds_y']])
        matrix1 = np.array(matrix1)

        matrix2 = []
        for i in Line2:
            matrix2.append([i['mds_x'], i['mds_y']])
        matrix2 = np.array(matrix2)

        matrix3 = []
        for i in Line3:
            matrix3.append([i['mds_x'], i['mds_y']])
        matrix3 = np.array(matrix3)

        matrix4 = []
        for i in Line4:
            matrix4.append([i['mds_x'], i['mds_y']])
        matrix4 = np.array(matrix4)


        mtx_r1 = np.array([matrix1[0], matrix1[12], matrix1[25], matrix1[37]])
        mtx_r2 = np.array([matrix2[0], matrix2[12], matrix2[25], matrix2[37]])
        mtx_r3 = np.array([matrix3[0], matrix3[12], matrix3[25], matrix3[37]])
        mtx_r4 = np.array([matrix4[0], matrix4[12], matrix4[25], matrix4[37]])

        mtx1 = matrix1*23
        R, s = orthogonal_procrustes(mtx_r1, mtx_r2)
        mtx2 = matrix2@R.T*23
        R, s = orthogonal_procrustes(mtx_r1, mtx_r3)
        mtx3 = matrix3@R.T*23
        R, s = orthogonal_procrustes(mtx_r1, mtx_r4)
        mtx4 = matrix4@R.T*23


        # std1 = np.std(mtx_r2)
        # std2 = np.std(mtx_r1)
#
#         std1 = np.std(mtx2)
#         std2 = np.std(mtx1)

        # Scalăm prima matrice la aceeași scară cu a doua matrice
#         mtx2 = mtx2 / std1 * std2

        for i in range(len(mtx1)):
            diff_lines.append(math.dist(mtx1[i], mtx2[i]))
            diff_lines.append(math.dist(mtx1[i], mtx3[i]))
            diff_lines.append(math.dist(mtx1[i], mtx4[i]))

    fig = plt.figure()
    sns.histplot(diff_lines, bins=int(max(diff_lines) - min(diff_lines)), kde=True, color='skyblue', edgecolor='black')
    print(int(max(diff_lines) - min(diff_lines)))
    print(len(diff_lines))
    print(f"max = {max(diff_lines)}")
    print(f"min = {min(diff_lines)}")
    print(f"avg = {np.average(diff_lines)}")

    # Adding labels and title
    plt.xlabel('Distance')
    plt.ylabel('Frequency')
    plt.title('Errors')

    # Display the plot
    plt.show()
    fig.savefig("../test-mds/images/raport-3/procrustes_histogram_all_floor_mds_data_andreea.png", bbox_inches='tight')