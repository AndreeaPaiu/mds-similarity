import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import euclidean_distances

from smacofAnchors.MDS_A import mds, plot_mds

ALGORITHMS = ('_smacof_with_anchors_single', '_smacof_with_anchors')
LABELS = dict(zip(ALGORITHMS, ['Anchored MDS','MDS-A']))

X_MAX = 30
X_MIN = 0
Y_MAX = 20
Y_MIN = 0

_ORIGIN = np.array([X_MIN, Y_MIN])
_ANCHORS = np.array([_ORIGIN,
			[X_MAX, Y_MIN],
			[X_MAX, Y_MAX],
			[X_MIN, Y_MAX],
			[(X_MAX+X_MIN)/2.0, Y_MAX],
			[(X_MAX+X_MIN)/2.0, Y_MIN],
			[X_MIN, (Y_MAX+Y_MIN)/2.0],
			[X_MAX, (Y_MAX+Y_MIN)/2.0]])

def generate_random_points(config):
    x0_displacements = np.random.choice(np.arange(1, X_MAX - X_MIN - 1), size=config['no_of_tags'])
    y0_displacements = np.random.choice(np.arange(1, Y_MAX - Y_MIN - 1), size=config['no_of_tags'])
    config['tags'] = np.array([_ORIGIN + pt for pt in zip(x0_displacements, y0_displacements)])
    config['points'] = np.concatenate((config['anchors'], config['tags']))

def set_anchors(config):
    config['anchors'] = _ANCHORS[:config['no_of_anchors']]

def rmse(computed, real):
    return np.sqrt(((computed - real)**2).mean())

def add_noise(cofing):
    distances = config['real_distance']
    if cofing['sigma'] > 0:
        noise = np.random.normal(config['mu'], config['sigma'], size=(config['no_of_anchors'], cofing['no_of_tags']))
        distances[:config['no_of_anchors'], -cofing['no_of_tags']:] += noise
        distances[-cofing['no_of_tags']:, :config['no_of_anchors']] += noise.T

        if config['missingdata']:
            # add noise to within-sets data when available
            noise = np.random.normal(config['mu'], config['sigma'], size=(config['no_of_tags'], config['no_of_tags']))
            distances[-config['no_of_tags']:, -config['no_of_tags']:] += (noise + noise.T) / 2.0
            # self distance should remain zero
            distances[np.arange(len(distances)), np.arange(len(distances))] = 0

    config['distance'] = distances

def estimate_coords(config):
    mds_ = mds.MDS(config, config['algorithm'], n_components=config['n_components'], max_iter=config['max_iter'], eps=config['eps'], random_state=config['random_state'], n_jobs=config['n_jobs'],
                   dissimilarity="precomputed", metric=True, verbose=0)  # n_init=1
    results = mds_.fit(config['distance'])
    return results.embedding_

def best_similarity_transform(X, Y):
    K = X.shape[0]
    Xm = X.mean(axis=1).reshape(K, 1)
    Ym = Y.mean(axis=1).reshape(K, 1)
    X = X - Xm
    Y = Y - Ym
    normX = np.linalg.norm(X)
    normY = np.linalg.norm(Y)
    X /= normX
    Y /= normY

    YX = np.dot(Y, X.T)
    U, S, V = np.linalg.svd(YX)
    R = np.dot(U, V)
    t = Ym - np.dot(R, Xm)
    s = S.sum() * normX / normY
    return R, t, s

def apply_rotation(cofing):
    Rotation, t, s = best_similarity_transform(config['distance'][:-config['no_of_tags'], :].T, config['real_distance'][:-config['no_of_tags'], :].T)
    config['distance'] = (Rotation.dot((config['distance'] / s).T) + t).T

def rmse_vs_noise(config):
    errors = []
    sigmas = np.linspace(0, 4, 40)
    for sigma in sigmas:
        config['sigma'] = sigma
        for trial in range(config['no_of_trials']):
            generate_random_points(config)
            config['real_distance'] = euclidean_distances(config['points'])
            if config['missingdata']:
                config['real_distance'][-config['no_of_tags']:, -config['no_of_tags']:] = 0
            add_noise(config)
            estimate_coords(config)
            apply_rotation(config)
            error = [rmse(config['real_distance'], config['distance']) for i in range(config['no_of_trials'])]
            errors.append(error)

    return sigmas, errors
def first_third_quartile_and_median(data):
    first_quartile = np.percentile(data, 25, axis=1)
    third_quartile = np.percentile(data, 75, axis=1)
    median = np.median(data, axis=1)
    return first_quartile, median, third_quartile

def plot_experiment(config):
    sigmas, errors = rmse_vs_noise(config)
    first_q, median, third_q = first_third_quartile_and_median(errors)
    plot_mds.plot_rmse_vs_noise(first_q, median, third_q)
    plot_hanldes = []
    plot_labels = []

    handles = plot_mds.plot_rmse_vs_noise(first_q, median, third_q, x_axis=sigmas)
    plot_hanldes.extend(handles)
    plot_labels.extend([LABELS[config['algorithm']], 'IQR boundaries'])
    plt.legend(loc='upper left', fontsize=12, labels=plot_labels, handles=plot_hanldes)
    plt.show()
    plt.legend(loc='upper left', fontsize=12, labels=plot_labels, handles=plot_hanldes)
    plt.show()

def run(config):
    for i in ALGORITHMS:
        config['algorithm'] = i

    plot_experiment(config)

if __name__ == '__main__':
    # set cofiguration
    config = {
        'no_of_tags': 30,
        'no_of_anchors': 4,
        'no_of_trials': 1,
        'missingdata': False,
        'sigma': 0,
        'mu': 0,
        'noise': 2,
        'algorithm': '_smacof_with_anchors',
        'n_components': 2,
        'eps': 1e-14,
        'max_iter': 500,
        'random_state': np.random.RandomState(seed=3),
        'n_jobs': 1,
    }
    set_anchors(config)
    run(config)

