import math

import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from sklearn.manifold import MDS
import matplotlib.patches as mpatches


def plot_mds_3D_real_cartesian_system(collections):
    # Matricea de similarități între produse
    similarities = np.empty((len(collections), len(collections)))
    count_floors = {}

    for i in range(len(collections)):
        if int(collections[i]['floor']) not in count_floors:
            count_floors[int(collections[i]['floor'])] = 0
        count_floors[int(collections[i]['floor'])] += 1
        for j in range(len(collections)):
            if 'real_coordinates' not in collections[i] or 'real_coordinates' not in collections[j]:
                similarities[i][j] = 100
                continue

            similarities[i, j] = math.dist(collections[i]['real_coordinates'][0:3], collections[j]['real_coordinates'][0:3])

    # Crearea unui obiect MDS cu 3 dimensiuni
    mds = MDS(n_components=3, dissimilarity='precomputed')

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

    # Vizualizarea rezultatelor
    fig = plt.figure()
    plt.axis('equal')
    ax = fig.add_subplot(111, projection='3d')
    lines = {}

    # Perform linear regression to fit a line to the points
    for i in floors:
        A = []
        A = np.column_stack((x[i], y[i], np.ones_like(x[i])))
        result = np.linalg.lstsq(A, z[i], rcond=None)
        lines[i] = result[0]
        print(f"Slope = {lines[i][:2]}, Intercept = {lines[i][2]}, Floor = {i}")
        # Generate points for the line of best fit
        x_line, y_line, z_line = [], [], []
        x_line = np.linspace(min(x[i]), max(x[i]), 100)
        y_line = np.linspace(min(y[i]), max(y[i]), 100)
        x_line, y_line = np.meshgrid(x_line, y_line)
        z_line = lines[i][0] * x_line + lines[i][1] * y_line + lines[i][2]
        # Plot the line of best fit in 3D
        ax.plot_surface(x_line, y_line, z_line, alpha=0.2)

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


    colors = []
    for color in matplotlib.colors.TABLEAU_COLORS:
        colors.append(color)

    for i, (x, y, z) in enumerate(coordinates):
        ax.scatter(x, y, z, color=colors[int(collections[i]['floor'])], label=f'Line {i+1}')
        ax.text(x + .03, y + .03, z + .03, i % count_floors[int(collections[i]['floor'])], fontsize=9)



    ax.set_xlabel('Dimensiune 1 (m)')
    ax.set_ylabel('Dimensiune 2 (m)')
    ax.set_zlabel('Dimensiune 3 (m)')
    plt.title('Reprezentarea Carteziana a punctelor folosind MDS')

    handles = []
    for i in range(0, int(collections[len(collections) - 1]['floor']) + 1):
        handles.append(mpatches.Patch(color=colors[i], label='etaj' + str(i)))

    plt.legend(handles=handles)
    plt.grid()
    plt.show()
    # fig.savefig(f"images/mds-cartesian_system.svg", bbox_inches='tight')
    # fig.savefig(f"images/mds-cartesian_system-all-floor.svg", bbox_inches='tight')
    fig.savefig(f"images/mds-cartesian_system-2-floors.svg", bbox_inches='tight')


