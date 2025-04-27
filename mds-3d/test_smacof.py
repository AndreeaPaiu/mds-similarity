import numpy as np
import networkx as nx
from sklearn.manifold import MDS
import matplotlib.pyplot as plt
from scipy.linalg import orthogonal_procrustes


def mds_map(graph, known_positions=None, dim=2, seed=42, align=True):
    """
    MDS-MAP: compute node positions via classical MDS on all-pairs shortest paths,
    optionally aligning to known anchor positions.

    Parameters:
    - graph: networkx.Graph with 'weight' on edges
    - known_positions: dict {node: [x, y, ...]} anchors for Procrustes alignment
    - dim: embedding dimensionality
    - seed: random seed for MDS init
    - align: whether to align to anchors

    Returns:
    - coords: dict {node: np.array(coordinates)} before alignment
    - aligned_coords: dict {node: np.array(coordinates)} after alignment (or same if align=False)
    """
    # List nodes and build index
    nodes = list(graph.nodes)
    n = len(nodes)
    idx = {node: i for i, node in enumerate(nodes)}

    # Compute shortest-path distance matrix
    D = np.zeros((n, n))
    sp = dict(nx.all_pairs_dijkstra_path_length(graph, weight='weight'))
    for u in nodes:
        for v in nodes:
            D[idx[u], idx[v]] = sp[u].get(v, np.inf)

    # Classical MDS
    mds = MDS(n_components=dim, dissimilarity='precomputed', random_state=seed)
    X = mds.fit_transform(D)
    coords = {node: X[idx[node]] for node in nodes}

    if align and known_positions:
        # Prepare matrices for anchors
        anchor_nodes = list(known_positions.keys())
        A = np.array([coords[n] for n in anchor_nodes])
        B = np.array([known_positions[n] for n in anchor_nodes])
        # Orthogonal Procrustes (rotation+scale)
        R, scale = orthogonal_procrustes(A, B)
        X_aligned = X @ R * scale
        aligned_coords = {node: X_aligned[idx[node]] for node in nodes}
        return coords, aligned_coords

    return coords, coords


# === Example Usage ===
if __name__ == "__main__":
    # True positions for simulation
    true_positions = {
        0: [0, 0],
        1: [1, 0],
        2: [1, 1],
        3: [0, 1],
        4: [0.5, 1.5]
    }
    # Build graph with edges for neighbors within threshold distance
    G = nx.Graph()
    for u, pu in true_positions.items():
        G.add_node(u)
    for u, pu in true_positions.items():
        for v, pv in true_positions.items():
            if u < v:
                dist = np.linalg.norm(np.array(pu) - np.array(pv))
                if dist < 1.2:  # connectivity threshold
                    G.add_edge(u, v, weight=dist)

    # Use nodes 0 and 2 as anchors
    anchors = {0: true_positions[0], 2: true_positions[2]}

    # Run MDS-MAP
    coords, aligned_coords = mds_map(G, known_positions=anchors)

    # Plot True vs Estimated
    plt.figure(figsize=(6, 6))
    true = np.array([true_positions[n] for n in G.nodes])
    est = np.array([aligned_coords[n] for n in G.nodes])
    plt.scatter(true[:, 0], true[:, 1], c='blue', label='True', s=50)
    plt.scatter(est[:, 0], est[:, 1], c='red', marker='x', label='Estimated', s=50)
    for i, (t, e) in enumerate(zip(true, est)):
        plt.plot([t[0], e[0]], [t[1], e[1]], 'k--', alpha=0.5)
        plt.text(t[0], t[1], str(i), color='blue', fontsize=9)
        plt.text(e[0], e[1], str(i), color='red', fontsize=9)
    plt.legend()
    plt.title("MDS-MAP: True vs Estimated Node Positions")
    plt.axis('equal')
    plt.grid(True)
    plt.show()
