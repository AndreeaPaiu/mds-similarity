import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches



def plot_real_cartesian_system(collection_points):
    fig = plt.figure()
    # plt.title('Real Cartesian System')
    # plt.xlabel('x')
    # plt.ylabel('y')

    colors = []
    for color in matplotlib.colors.TABLEAU_COLORS:
        colors.append(color)
    # for i in range(len(collection_points)):
    #     if 'real_coordinates' in collection_points[i]:
    #         plt.plot(
    #             collection_points[i]['real_coordinates'][0],
    #             collection_points[i]['real_coordinates'][1],
    #             color=colors[int(collection_points[i]['floor'])]
    #         )
    #         plt.text(collection_points[i]['real_coordinates'][0] + .03, collection_points[i]['real_coordinates'][1] + .03, i, fontsize=9)

    ax = fig.add_subplot(111, projection='3d')

    count_floors = {}
    for collection in collection_points:
        if int(collection['floor']) not in count_floors:
            count_floors[int(collection['floor'])] = 0
        count_floors[int(collection['floor'])] += 1

    for i in range(len(collection_points)):
        collection = collection_points[i]
        print('here')
        if 'real_coordinates' in collection:
            ax.scatter(
                collection['real_coordinates'][0],
                collection['real_coordinates'][1],
                collection['real_coordinates'][2],
                color=colors[int(collection['floor'])]
            )
            ax.text(collection['real_coordinates'][0],
                    collection['real_coordinates'][1],
                    collection['real_coordinates'][2],
                    i % count_floors[int(collection['floor'])],
                    fontsize=9
                    )

    # x = {}
    # y = {}
    # z = {}
    # floors = []
    # for r in range(len(collection_points)):
    #     if int(collection_points[r]['floor']) not in x:
    #         floors.append(int(collection_points[r]['floor']))
    #         x[int(collection_points[r]['floor'])] = []
    #     if int(collection_points[r]['floor']) not in y:
    #         y[int(collection_points[r]['floor'])] = []
    #     if int(collection_points[r]['floor']) not in z:
    #         z[int(collection_points[r]['floor'])] = []
    #
    #     x[int(collection_points[r]['floor'])].append(collection_points[r]['real_coordinates'][0])
    #     y[int(collection_points[r]['floor'])].append(collection_points[r]['real_coordinates'][1])
    #     z[int(collection_points[r]['floor'])].append(collection_points[r]['real_coordinates'][2])
    #
    # # Perform linear regression to fit a line to the points
    # for i in floors:
    #     A = []
    #     A = np.column_stack((x[i], y[i], np.ones_like(x[i])))
    #     result = np.linalg.lstsq(A, z[i], rcond=None)
    #     line = []
    #     line = result[0]
    #     print(f"Slope = {line[:2]}, Intercept = {line[2]}")
    #     # Generate points for the line of best fit
    #     x_line, y_line, z_line = [], [], []
    #     x_line = np.linspace(min(x[i]), max(x[i]), 100)
    #     y_line = np.linspace(min(y[i]), max(y[i]), 100)
    #     x_line, y_line = np.meshgrid(x_line, y_line)
    #     z_line = line[0] * x_line + line[1] * y_line + line[2]
    #     # Plot the line of best fit in 3D
    #     ax.plot_surface(x_line, y_line, z_line, alpha=0.2)
    #
    # ax.set_xlabel('x (m)')
    # ax.set_ylabel('y (m)')
    # ax.set_zlabel('z (m)')
    # handles = []
    # for i in range(0, int(collection_points[len(collection_points) - 1]['floor']) + 1):
    #     handles.append(mpatches.Patch(color=colors[i], label='etaj' + str(i)))

    # plt.legend(handles=handles)
    plt.grid()
    plt.show()
    # fig.savefig(f"images/label-real-cartesian_system.svg", bbox_inches='tight')
    fig.savefig(f"images/real-cartesian_system-all-floors.svg", bbox_inches='tight')
