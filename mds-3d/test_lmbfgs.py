import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.spatial.distance import cdist, pdist, squareform
from scipy.linalg import orthogonal_procrustes
from sklearn.metrics import euclidean_distances

class PointReconstructor:
    def __init__(self, D, known, init_embeddings=None):
        """
        D: full dissimilarity (distance) matrix of shape (n_total, n_total)
        known: array of shape (n_known, 2) for the first n_known true points
        init_embeddings: optional array of shape (n_total, 2) containing an initial normalized
                         embedding for all points; if provided, will be aligned to known true points
        """
        self.D = D
        self.known = known
        self.n_known = known.shape[0]
        self.n_total = D.shape[0]
        self.n_unknown = self.n_total - self.n_known
        self.init_embeddings = init_embeddings
        self.estimated = None

    def smart_init(self):
        """Fallback initial guess if no init_embeddings provided"""
        if self.n_known == 0:
            # random in [-10, 10]
            return (np.random.rand(self.n_unknown * 2) * 20) - 10
        guesses = []
        for i in range(self.n_known, self.n_total):
            dists = self.D[i, :self.n_known]
            j = np.argmin(dists)
            pt = self.known[j]
            guesses.append(pt + np.array([dists[j], 0]))
        return np.array(guesses).flatten()

    def align_scale(self):
        """
        Align and scale init_embeddings so that first n_known match known true points via
        a similarity transform (scale + rotation + translation).
        Returns: mapped array of shape (n_total, 2)
        """
        X_all = self.init_embeddings
        # split known and all
        X_known = X_all[:self.n_known]
        Y_true = self.known
        # centroids
        mu_X = X_known.mean(axis=0)
        mu_Y = Y_true.mean(axis=0)
        # center
        A = X_known - mu_X
        B = Y_true  - mu_Y
        # orthogonal Procrustes gives R and scale s
        R, s = orthogonal_procrustes(A, B)
        # apply to all
        Xc = X_all - mu_X
        mapped = (s * Xc.dot(R)) + mu_Y
        return mapped

    def get_init_guess(self):
        """
        Compute the flattened initial guess for unknown points:
        - if init_embeddings provided: align+scale then take the unknown part
        - else: smart_init()
        """
        if self.init_embeddings is not None:
            aligned = self.align_scale()
            return aligned[self.n_known:].flatten()
        return self.smart_init()

    def stress(self, unknown_flat):
        """Objective: weighted stress of unknown + known vs D"""
        unknown = unknown_flat.reshape(self.n_unknown, 2)
        pts = unknown if self.n_known == 0 else np.vstack([self.known, unknown])
        dist_mat = euclidean_distances(pts)
        return np.nansum((dist_mat.ravel() - self.D.ravel())**2) / 2

    def reconstruct(self):
        """Optimizes to find unknown point positions."""
        if self.n_unknown <= 0:
            raise ValueError("No unknown points to reconstruct.")
        init_guess = self.get_init_guess()
        res = minimize(self.stress, init_guess, method='L-BFGS-B', tol=1e-15)
        self.estimated = res.x.reshape(self.n_unknown, 2)
        return self.estimated

    def all_points(self):
        """Return stacked known + estimated"""
        if self.estimated is None:
            raise ValueError("Call reconstruct() before accessing all_points.")
        return np.vstack([self.known, self.estimated])

    def stress_value(self):
        """Final stress after reconstruction"""
        pts = self.all_points()
        return np.nansum((euclidean_distances(pts).ravel() - self.D.ravel())**2) / 2

    def plot(self):
        """Plot known (red) and estimated (blue) points."""
        pts = self.all_points()
        plt.figure(figsize=(8, 6))
        plt.scatter(pts[:,0], pts[:,1], c='blue', label='All Points')
        if self.n_known > 0:
            plt.scatter(self.known[:,0], self.known[:,1], c='red', label='Known Points', zorder=5)
        for i, (x,y) in enumerate(self.known):
            plt.text(x, y, f'P{i+1}', fontsize=12, ha='right', color='red')
        for i, (x,y) in enumerate(self.estimated):
            plt.text(x, y, f'P{i+self.n_known+1}', fontsize=12, ha='left', color='blue')
        plt.title("Reconstructed Points from Dissimilarities")
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.legend()
        plt.grid(True)
        plt.show()
