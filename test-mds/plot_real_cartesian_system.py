import matplotlib.pyplot as plt


def plot_real_cartesian_system(collection_points):
    fig = plt.figure()
    plt.title('Real Cartesian System')
    plt.xlabel('x')
    plt.ylabel('y')


    for collection in collection_points:
        if 'real_coordinates' in collection:
            plt.plot(
                collection['real_coordinates'][0],
                collection['real_coordinates'][1],
                'o'
            )

    plt.show()
    fig.savefig(f"real-cartesian_system.pdf", bbox_inches='tight')
