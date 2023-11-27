import matplotlib.pyplot as plt


def plot_real_cartesian_system(collection_points):
    plt.title('Real Cartesian System')
    plt.xlabel('x')
    plt.ylabel('y')

    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
    fig = plt.figure()

    for collection in collection_points:
        if 'real_coordinates' in collection:
            plt.plot(
                collection['real_coordinates'][0],
                collection['real_coordinates'][1],
                colors[int(collection['floor'])] + 'o'
            )

    # ax = fig.add_subplot(111, projection='3d')

    # for collection in collection_points:
    #     if 'real_coordinates' in collection:
    #         ax.plot(
    #             collection['real_coordinates'][0],
    #             collection['real_coordinates'][1],
    #             collection['real_coordinates'][2],
    #             colors[int(collection['floor'])] + 'o'
    #         )

    plt.show()
    fig.savefig(f"images/real-cartesian_system.svg", bbox_inches='tight')
    # fig.savefig(f"images/real-cartesian_system-all-floors.svg", bbox_inches='tight')
