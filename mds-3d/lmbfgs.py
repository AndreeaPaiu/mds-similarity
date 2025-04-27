import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from sklearn.metrics import euclidean_distances
from scipy.linalg import orthogonal_procrustes

class PointReconstructor:
    def __init__(self, D, known, n_components=2, apply_scaling=True):
        """
        D: full dissimilarity (distance) matrix, shape (n_total, n_total)
        known: array of shape (n_known, 2) for the true coords of first n_known points
        n_components: target dimension (default=2)
        apply_scaling: if True, estimate and apply a global scale based on known-known distances
        """
        self.D = np.asarray(D, float)
        self.known = np.asarray(known, float).reshape(-1, n_components)
        self.n_known = self.known.shape[0]
        self.n_total = self.D.shape[0]
        self.n_unknown = self.n_total - self.n_known
        self.n_components = n_components
        # optionally scale D to best match the true known distances
        if apply_scaling and self.n_known > 1:
            scale = self.estimate_scale_factor()
            self.D *= scale
        self.estimated = None

    @staticmethod
    def classical_mds(D, n_components=2):
        D = np.asarray(D, float)
        D2 = D ** 2
        n = D2.shape[0]
        J = np.eye(n) - np.ones((n, n)) / n
        B = -0.5 * J.dot(D2).dot(J)
        eigvals, eigvecs = np.linalg.eigh(B)
        idx = np.argsort(eigvals)[::-1]
        L = np.diag(np.sqrt(np.maximum(eigvals[idx[:n_components]], 0)))
        V = eigvecs[:, idx[:n_components]]
        return V.dot(L)

    def estimate_scale_factor(self):
        """
        Estimate a global scale alpha to best align D[i,j] to actual distances
        among known points. Solves min_alpha sum((alpha*D_ij - d_ij)^2).
        """
        # collect known-known pairs
        pairs = []
        dists_D = []
        dists_true = []
        for i in range(self.n_known):
            for j in range(i+1, self.n_known):
                dists_D.append(self.D[i, j])
                dists_true.append(np.linalg.norm(self.known[i] - self.known[j]))
        dists_D = np.array(dists_D)
        dists_true = np.array(dists_true)
        # least-squares estimate of alpha: (d⋅δ)/(δ⋅δ)
        alpha = np.dot(dists_true, dists_D) / np.dot(dists_D, dists_D)
        return alpha

    def align_scale(self, X_all):
        if self.n_known == 0:
            return X_all
        X_known = X_all[:self.n_known]
        Y_true = self.known
        mu_X = X_known.mean(axis=0)
        mu_Y = Y_true.mean(axis=0)
        A = X_known - mu_X
        B = Y_true - mu_Y
        R, s = orthogonal_procrustes(A, B)
        Xc = X_all - mu_X
        return (s * Xc.dot(R)) + mu_Y

    def get_init_guess(self):
        embed = self.classical_mds(self.D, n_components=self.n_components)
        aligned = self.align_scale(embed)
        if self.n_known == 0:
            unk = aligned
        else:
            unk = aligned[self.n_known:]
        return unk.flatten()

    def stress(self, unknown_flat):
        unknown = unknown_flat.reshape(self.n_unknown, self.n_components)
        pts = unknown if self.n_known == 0 else np.vstack([self.known, unknown])
        dist_mat = euclidean_distances(pts)
        return np.sum((dist_mat.ravel() - self.D.ravel()) ** 2) / 2

    def reconstruct(self, tol=1e-12, method='L-BFGS-B'):
        if self.n_unknown <= 0:
            raise ValueError("No unknown points to reconstruct.")
        init_guess = self.get_init_guess()
        res = minimize(self.stress, init_guess, method=method, tol=tol)
        self.estimated = res.x.reshape(self.n_unknown, self.n_components)
        return self.estimated

    def all_points(self):
        if self.estimated is None:
            raise ValueError("Call reconstruct() before accessing all_points.")
        return np.vstack([self.known, self.estimated]) if self.n_known else self.estimated

    def stress_value(self):
        pts = self.all_points()
        return np.sum((euclidean_distances(pts).ravel() - self.D.ravel()) ** 2) / 2

    def plot(self):
        pts = self.all_points()
        plt.figure(figsize=(8, 6))
        plt.scatter(pts[:,0], pts[:,1], c='blue', label='All Points')
        if self.n_known:
            plt.scatter(self.known[:,0], self.known[:,1], c='red', label='Known Points', zorder=5)
        for i, (x, y) in enumerate(self.known):
            plt.text(x, y, f'P{i+1}', fontsize=12, ha='right', color='red')
        for i, (x, y) in enumerate(self.estimated):
            plt.text(x, y, f'P{i+self.n_known+1}', fontsize=12, ha='left', color='blue')
        plt.title("Reconstructed Points from Dissimilarities via MDS + Procrustes")
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.legend()
        plt.grid(True)
        plt.show()