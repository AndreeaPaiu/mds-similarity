from __future__ import division

import operator

import numpy as np

import warnings

from sklearn.base import BaseEstimator
from sklearn.metrics import euclidean_distances
from sklearn.utils import check_random_state, check_array, check_symmetric
from joblib import Parallel
from joblib import delayed
from sklearn.isotonic import IsotonicRegression

def _smacof_with_anchors_single(config, similarities, metric=True, n_components=2, init=None,
                                max_iter=300, verbose=0, eps=1e-3, random_state=None):

    similarities = check_symmetric(similarities, raise_exception=True)

    n_samples = similarities.shape[0]
    random_state = check_random_state(random_state)

    sim_flat = ((1 - np.tri(n_samples)) * similarities).ravel()
    sim_flat_w = sim_flat[sim_flat != 0]

    if init is None:
        # Randomly choose initial configuration
        X = random_state.rand(n_samples * n_components)
        X = X.reshape((n_samples, n_components))
    # uncomment the following if weight matrix W is not hollow
    else:
        # overrides the parameter p
        n_components = init.shape[1]
        if n_samples != init.shape[0]:
            raise ValueError("init matrix should be of shape (%d, %d)" %
                             (n_samples, n_components))
        X = init

    old_stress = None
    ir = IsotonicRegression()

    # setup weight matrix
    weights = np.ones((n_samples, n_samples))
    if config.missingData:
        weights[-config.no_of_tags:, -config.no_of_tags:] = 0

    diag = np.arange(n_samples)
    weights[diag, diag] = 0

    last_n_configs = []
    Xa = config.anchors
    for it in range(max_iter):
        # Compute distance and monotonic regression
        dis = euclidean_distances(X)

        if metric:
            disparities = similarities
        else:
            dis_flat = dis.ravel()
            # similarities with 0 are considered as missing values
            dis_flat_w = dis_flat[sim_flat != 0]

            # Compute the disparities using a monotonic regression
            disparities_flat = ir.fit_transform(sim_flat_w, dis_flat_w)
            disparities = dis_flat.copy()
            disparities[sim_flat != 0] = disparities_flat
            disparities = disparities.reshape((n_samples, n_samples))
            disparities *= np.sqrt((n_samples * (n_samples - 1) / 2) /
                                   (disparities ** 2).sum())

        # Compute stress
        stress = (weights.ravel() * (dis.ravel() - disparities.ravel()) ** 2).sum() / 2

        # Update X using the Guttman transform
        dis[dis == 0] = 1e5
        ratio = weights * disparities / dis
        B = - ratio
        B[diag, diag] = 0
        B[diag, diag] = -B.sum(axis=1)

        # Apply update to only tag configuration since anchor config is already known

        V = - weights
        V[diag, diag] += weights.sum(axis=1)
        V12 = V[-config.no_of_tags:, :-config.no_of_tags]
        B11 = B[-config.no_of_tags:, -config.no_of_tags:]
        Zu = X[-config.no_of_tags:]
        B12 = B[-config.no_of_tags:, :-config.no_of_tags]
        V11_inv = np.linalg.inv(V[-config.no_of_tags:, -config.no_of_tags:])
        Xu = V11_inv.dot(B11.dot(Zu) + (B12 - V12).dot(Xa))

        # merge known anchors config with new tags config 
        X = np.concatenate((Xa, Xu))
        last_n_configs.append(X)

        dis = (weights * dis ** 2).sum() / 2
        if verbose >= 2:
            print('it: %d, stress %s' % (it, stress))
        if old_stress is not None:
            if (old_stress - stress / dis) < eps:
                if verbose:
                    print('breaking at iteration %d with stress %s' % (it,
                                                                       stress))
                break
        old_stress = stress / dis
    return X, stress, it + 1, np.array(last_n_configs)


def compute_V(W):
    nodes = W.shape[0]
    if W is None:
        W = np.ones((nodes, nodes)) - np.eye(nodes)
    return np.diag(np.sum(W, axis=1)) - W


def _smacof_with_anchors(config, similarities, metric=True, n_components=2, init=None,
                         max_iter=300, verbose=0, eps=1e-3, random_state=None):
    similarities = check_symmetric(similarities, raise_exception=True)

    n_samples = similarities.shape[0]
    random_state = check_random_state(random_state)

    if init is None:
        # Randomly choose initial configuration
        X = random_state.rand(n_samples * n_components)
        X = X.reshape((n_samples, n_components))
        X = X * config.x_max

        X[:-config.no_of_tags, :] = config.anchors
        # ---------------------
    # uncomment the following if weight matrix W is not hollow
    # X[:-2] = Xa
    else:
        # overrides the parameter p
        n_components = init.shape[1]
        if n_samples != init.shape[0]:
            raise ValueError("init matrix should be of shape (%d, %d)" %
                             (n_samples, n_components))
        X = init

    old_stress = None
    ir = IsotonicRegression()

    # setup weight matrix
    weights = np.ones((n_samples, n_samples))
    if hasattr(config, 'weight_matrix') and config.weight_matrix is not None:
        # Folosim matricea personalizată unde 0 înseamnă "distanță lipsă" (D=1.0)
        weights = config.weight_matrix.copy()
    else:
        weights = np.ones((n_samples, n_samples))
        if config.missingData:
            weights[-config.no_of_tags:, -config.no_of_tags:] = 0

    diag = np.arange(n_samples)
    weights[diag, diag] = 0

    last_n_configs = []
    Xa = config.anchors
    for it in range(max_iter):
        # Compute distance and monotonic regression
        dis = euclidean_distances(X)

        disparities = similarities

        # Compute stress
        stress = (weights.ravel() * (dis.ravel() - disparities.ravel()) ** 2).sum() / 2

        # Update X using the Guttman transform
        dis[dis == 0] = 1e5
        ratio = weights * disparities / dis
        B = np.diag(ratio.sum(axis=1)) - ratio

        # Apply update to only tag configuration since anchor config is already known

        V = compute_V(weights)
        V12 = V[-config.no_of_tags:, :-config.no_of_tags]
        B11 = B[-config.no_of_tags:, -config.no_of_tags:]
        Zu = X[-config.no_of_tags:]
        Za = X[:-config.no_of_tags, :]
        B12 = B[-config.no_of_tags:, :-config.no_of_tags]
        V11_inv = np.linalg.inv(V[-config.no_of_tags:, -config.no_of_tags:])
        Xu = V11_inv.dot(B11.dot(Zu) + B12.dot(Za) - V12.dot(Xa))

        # merge known anchors config with new tags config
        X = np.concatenate((Xa, Xu))
        last_n_configs.append(X)

        dis = (weights * dis ** 2).sum() / 2
        if verbose >= 2:
            print('it: %d, stress %s' % (it, stress))
        if old_stress is not None:
            if (old_stress - stress / dis) < eps:
                if verbose:
                    print('breaking at iteration %d with stress %s' % (it,
                                                                       stress))
                break
        old_stress = stress / dis
    return X, stress, it + 1, np.array(last_n_configs)


def _smacof_single(config, similarities, metric=True, n_components=2, init=None,
                   max_iter=300, verbose=0, eps=1e-3, random_state=None, estimated_dist_weights=0):
    similarities = check_symmetric(similarities, raise_exception=True)

    n_samples = similarities.shape[0]
    random_state = check_random_state(random_state)

    sim_flat = ((1 - np.tri(n_samples)) * similarities).ravel()
    sim_flat_w = sim_flat[sim_flat != 0]
    if init is None:
        # Randomly choose initial configuration
        X = random_state.rand(n_samples * n_components)
        X = X.reshape((n_samples, n_components))
    else:
        # overrides the parameter p
        n_components = init.shape[1]
        if n_samples != init.shape[0]:
            raise ValueError("init matrix should be of shape (%d, %d)" %
                             (n_samples, n_components))
        X = init

    old_stress = None
    ir = IsotonicRegression()

    # setup weight matrix
    weights = np.ones((n_samples, n_samples))
    weights[-config.no_of_tags:, -config.no_of_tags:] = estimated_dist_weights
    diag = np.arange(n_samples)
    weights[diag, diag] = 0

    last_n_configs = []
    for it in range(max_iter):
        # Compute distance and monotonic regression
        dis = euclidean_distances(X)

        if metric:
            disparities = similarities
        else:
            dis_flat = dis.ravel()
            # similarities with 0 are considered as missing values
            dis_flat_w = dis_flat[sim_flat != 0]

            # Compute the disparities using a monotonic regression
            disparities_flat = ir.fit_transform(sim_flat_w, dis_flat_w)
            disparities = dis_flat.copy()
            disparities[sim_flat != 0] = disparities_flat
            disparities = disparities.reshape((n_samples, n_samples))
            disparities *= np.sqrt((n_samples * (n_samples - 1) / 2) /
                                   (disparities ** 2).sum())

        # Compute stress
        stress = (weights.ravel() * (dis.ravel() - disparities.ravel()) ** 2).sum() / 2

        # Update X using the Guttman transform
        dis[dis == 0] = 1e5
        ratio = disparities / dis
        B = - ratio

        B[diag, diag] += ratio.sum(axis=1)

        # Apply update to only tag configuration since anchor config is already known

        V = - weights
        V[diag, diag] += weights.sum(axis=1)
        V_inv = np.linalg.pinv(V)
        X = V_inv.dot(np.dot(B, X))

        last_n_configs.append(X)

        stress = (weights.ravel() * (dis.ravel() - disparities.ravel()) ** 2).sum() / 2
        dis[np.arange(n_samples), np.arange(n_samples)] = 0
        dis = (weights * dis ** 2).sum() / 2

        # dis = np.sqrt((X ** 2).sum(axis=1)).sum()
        if verbose >= 2:
            print('it: %d, stress %s' % (it, stress))
        if old_stress is not None:
            if (old_stress - stress / dis) < eps:
                if verbose:
                    print('breaking at iteration %d with stress %s' % (it,
                                                                       stress))
                break
        old_stress = stress / dis

    return X, stress, it + 1, np.array(last_n_configs)


VARIANTS = {'_smacof_with_anchors_single': _smacof_with_anchors_single,
            '_smacof_single': _smacof_single,
            '_smacof_with_anchors': _smacof_with_anchors}


def smacof(config, variant, similarities, metric=True, n_components=2, init=None, n_init=8,
           n_jobs=1, max_iter=300, verbose=0, eps=1e-3, random_state=None,
           return_n_iter=False):
    """
    Computes multidimensional scaling using SMACOF (Scaling by Majorizing a
    Complicated Function) algorithm
    The SMACOF algorithm is a multidimensional scaling algorithm: it minimizes
    a objective function, the *stress*, using a majorization technique. The
    Stress Majorization, also known as the Guttman Transform, guarantees a
    monotone convergence of Stress, and is more powerful than traditional
    techniques such as gradient descent.
    The SMACOF algorithm for metric MDS can summarized by the following steps:
    1. Set an initial start configuration, randomly or not.
    2. Compute the stress
    3. Compute the Guttman Transform
    4. Iterate 2 and 3 until convergence.
    The nonmetric algorithm adds a monotonic regression steps before computing
    the stress.
    Parameters
    ----------
    config : Config object
        configuration object for anchor-tag deployment parameters
    variant :  str
        variant of MDS algorithm to be used for computing configuration
    similarities : symmetric ndarray, shape (n_samples, n_samples)
        similarities between the points
    metric : boolean, optional, default: True
        compute metric or nonmetric SMACOF algorithm
    n_components : int, optional, default: 2
        number of dimension in which to immerse the similarities
        overridden if initial array is provided.
    init : {None or ndarray of shape (n_samples, n_components)}, optional
        if None, randomly chooses the initial configuration
        if ndarray, initialize the SMACOF algorithm with this array
    n_init : int, optional, default: 8
        Number of time the smacof algorithm will be run with different
        initialisation. The final results will be the best output of the
        n_init consecutive runs in terms of stress.
    n_jobs : int, optional, default: 1
        The number of jobs to use for the computation. This works by breaking
        down the pairwise matrix into n_jobs even slices and computing them in
        parallel.
        If -1 all CPUs are used. If 1 is given, no parallel computing code is
        used at all, which is useful for debugging. For n_jobs below -1,
        (n_cpus + 1 + n_jobs) are used. Thus for n_jobs = -2, all CPUs but one
        are used.
    max_iter : int, optional, default: 300
        Maximum number of iterations of the SMACOF algorithm for a single run
    verbose : int, optional, default: 0
        level of verbosity
    eps : float, optional, default: 1e-6
        relative tolerance w.r.t stress to declare converge
    random_state : integer or numpy.RandomState, optional
        The generator used to initialize the centers. If an integer is
        given, it fixes the seed. Defaults to the global numpy random
        number generator.
    return_n_iter : bool
        Whether or not to return the number of iterations.
    Returns
    -------
    X : ndarray (n_samples,n_components)
        Coordinates of the n_samples points in a n_components-space
    stress : float
        The final value of the stress (sum of squared distance of the
        disparities and the distances for all constrained points)
    n_iter : int
        The number of iterations corresponding to the best stress.
        Returned only if `return_n_iter` is set to True
    last_positions: ndarray [X1,...,Xn]
        An array of computed Xs from the selected mds/smacof variant, 
        used displaying trails showing convergence in animation.

    Notes
    -----
    "Modern Multidimensional Scaling - Theory and Applications" Borg, I.;
    Groenen P. Springer Series in Statistics (1997)
    "Nonmetric multidimensional scaling: a numerical method" Kruskal, J.
    Psychometrika, 29 (1964)
    "Multidimensional scaling by optimizing goodness of fit to a nonmetric
    hypothesis" Kruskal, J. Psychometrika, 29, (1964)
    """

    similarities = check_array(similarities)
    random_state = check_random_state(random_state)

    if hasattr(init, '__array__'):
        init = np.asarray(init).copy()
        if not n_init == 1:
            warnings.warn(
                'Explicit initial positions passed: '
                'performing only one init of the MDS instead of %d'
                % n_init)
            n_init = 1

    best_pos, best_stress = None, None
    smacof_variant = VARIANTS[variant]
    if n_jobs == 1:
        for it in range(n_init):
            pos, stress, n_iter_, last_n_pos = smacof_variant(
                config, similarities, metric=metric,
                n_components=n_components, init=init,
                max_iter=max_iter, verbose=verbose,
                eps=eps, random_state=random_state)
            if best_stress is None or stress < best_stress:
                best_stress = stress
                best_pos = pos.copy()
                best_iter = n_iter_
                best_last_n_pos = last_n_pos
    else:
        seeds = random_state.randint(np.iinfo(np.int32).max, size=n_init)
        results = Parallel(n_jobs=n_jobs, verbose=max(verbose - 1, 0))(
            delayed(smacof_variant)(
                config, similarities, metric=metric, n_components=n_components,
                init=init, max_iter=max_iter, verbose=verbose, eps=eps,
                random_state=seed)
            for seed in seeds)
        positions, stress, n_iters, last_n_pos = zip(*results)
        best = np.argmin(stress)
        best_stress = stress[best]
        best_pos = positions[best]
        best_iter = n_iters[best]
        best_last_n_pos = last_n_pos[best]

    if return_n_iter:
        return best_pos, best_stress, best_iter, best_last_n_pos
    else:
        return best_pos, best_stress, best_last_n_pos


class MDS(BaseEstimator):
    """Multidimensional scaling
    Read more in the :ref:`User Guide <multidimensional_scaling>`.
    Parameters
    ----------
    config : Config object
        configuration object for anchor-tag deployment parameters
    algorithm :  str
        MDS algorithm to be used for computing configuration
    metric : boolean, optional, default: True
        compute metric or nonmetric SMACOF (Scaling by Majorizing a
        Complicated Function) algorithm
    n_components : int, optional, default: 2
        number of dimension in which to immerse the similarities
        overridden if initial array is provided.
    n_init : int, optional, default: 4
        Number of time the smacof algorithm will be run with different
        initialisation. The final results will be the best output of the
        n_init consecutive runs in terms of stress.
    max_iter : int, optional, default: 300
        Maximum number of iterations of the SMACOF algorithm for a single run
    verbose : int, optional, default: 0
        level of verbosity
    eps : float, optional, default: 1e-6
        relative tolerance w.r.t stress to declare converge
    n_jobs : int, optional, default: 1
        The number of jobs to use for the computation. This works by breaking
        down the pairwise matrix into n_jobs even slices and computing them in
        parallel.
        If -1 all CPUs are used. If 1 is given, no parallel computing code is
        used at all, which is useful for debugging. For n_jobs below -1,
        (n_cpus + 1 + n_jobs) are used. Thus for n_jobs = -2, all CPUs but one
        are used.
    random_state : integer or numpy.RandomState, optional
        The generator used to initialize the centers. If an integer is
        given, it fixes the seed. Defaults to the global numpy random
        number generator.
    dissimilarity : string
        Which dissimilarity measure to use.
        Supported are 'euclidean' and 'precomputed'.
    Attributes
    ----------
    embedding_ : array-like, shape [n_components, n_samples]
        Stores the position of the dataset in the embedding space
    stress_ : float
        The final value of the stress (sum of squared distance of the
        disparities and the distances for all constrained points)
    References
    ----------
    "Modern Multidimensional Scaling - Theory and Applications" Borg, I.;
    Groenen P. Springer Series in Statistics (1997)
    "Nonmetric multidimensional scaling: a numerical method" Kruskal, J.
    Psychometrika, 29 (1964)
    "Multidimensional scaling by optimizing goodness of fit to a nonmetric
    hypothesis" Kruskal, J. Psychometrika, 29, (1964)
    """

    def __init__(self, config, algorithm, n_components=2, metric=True, n_init=4,
                 max_iter=300, verbose=0, eps=1e-3, n_jobs=1,
                 random_state=None, dissimilarity="euclidean"):
        self.n_components = n_components
        self.dissimilarity = dissimilarity
        self.metric = metric
        self.n_init = n_init
        self.max_iter = max_iter
        self.eps = eps
        self.verbose = verbose
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.variant = algorithm
        self.config = config

    @property
    def _pairwise(self):
        return self.kernel == "precomputed"

    def fit(self, X, y=None, init=None):
        """
        Computes the position of the points in the embedding space
        Parameters
        ----------
        X : array, shape=[n_samples, n_features], or [n_samples, n_samples] \
                if dissimilarity='precomputed'
            Input data.
        init : {None or ndarray, shape (n_samples,)}, optional
            If None, randomly chooses the initial configuration
            if ndarray, initialize the SMACOF algorithm with this array.
        """
        self.fit_transform(X, init=init)
        return self

    def fit_transform(self, X, y=None, init=None):
        """
        Fit the data from X, and returns the embedded coordinates
        Parameters
        ----------
        X : array, shape=[n_samples, n_features], or [n_samples, n_samples] \
                if dissimilarity='precomputed'
            Input data.
        init : {None or ndarray, shape (n_samples,)}, optional
            If None, randomly chooses the initial configuration
            if ndarray, initialize the SMACOF algorithm with this array.
        """
        X = check_array(X)
        if X.shape[0] == X.shape[1] and self.dissimilarity != "precomputed":
            warnings.warn("The MDS API has changed. ``fit`` now constructs an"
                          " dissimilarity matrix from data. To use a custom "
                          "dissimilarity matrix, set "
                          "``dissimilarity='precomputed'``.")

        if self.dissimilarity == "precomputed":
            self.dissimilarity_matrix_ = X
        elif self.dissimilarity == "euclidean":
            self.dissimilarity_matrix_ = euclidean_distances(X)
        else:
            raise ValueError("Proximity must be 'precomputed' or 'euclidean'."
                             " Got %s instead" % str(self.dissimilarity))

        self.embedding_, self.stress_, self.n_iter_, self.last_n_embeddings = smacof(self.config, self.variant,
                                                                                     self.dissimilarity_matrix_,
                                                                                     metric=self.metric,
                                                                                     n_components=self.n_components,
                                                                                     init=init, n_init=self.n_init,
                                                                                     n_jobs=self.n_jobs,
                                                                                     max_iter=self.max_iter,
                                                                                     verbose=self.verbose,
                                                                                     eps=self.eps,
                                                                                     random_state=self.random_state,
                                                                                     return_n_iter=True)

        return self.embedding_, self.last_n_embeddings