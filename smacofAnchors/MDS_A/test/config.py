import numpy as np
from scipy.spatial.distance import cosine
from sklearn.metrics import euclidean_distances

from test_mds.compute_mds_wifi_similarity import compute_mds_wifi_similarity

# ALGORITHMS = ('_smacof_with_anchors_single', '_smacof_with_anchors')
ALGORITHMS = ('_smacof_with_anchors_single','_smacof_with_anchors')
LABELS = dict(zip(ALGORITHMS, ['Anchored MDS', 'MDS-A']))
LABELS_MISSING = dict(zip((False, True), ['MDS-A', 'MDS-A missing data']))
# LABELS = dict(zip(ALGORITHMS, ['MDS-A']))

class SettingsManager:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not hasattr(self, '_initialized'):
            self.no_of_tags = 30
            self.no_of_anchors = 4
            self.x_max = 30
            self.y_max = 30
            self.x_min = 0
            self.y_min = 0
            self.z_max = 30
            self.z_min = 0
            self.no_of_trials = 100
            self.sigma = 0
            self.max_sigma = 4
            self.noise = 2
            self.missingData = False
            self.mu = 0
            self.algorithm = ALGORITHMS[0]
            self.n_components = 2
            self.eps = 1e-14
            self.max_iter = 500
            self.random_state = np.random.RandomState(seed=3)
            self.n_jobs = 1
            self.points = None
            self.anchors = None
            self.tags = None
            self._initialized = True
            self.true_distance = None
            self.random_anchors = False
            self.real_data = None
            self.is_random = False
            self.distance = None
            self.is_real_data = False
            self.missing_data_ratio = 0.0
            self.weight_matrix = None
            self.triang_dist = None
            self.initial_guess = None
            self.use_triangulation_init = None
            self.n_init = 10

    @property
    def no_of_points(self):
        return self.no_of_anchors + self.no_of_tags

    def update_settings(self, **kwargs):
        for key, value in kwargs.items():
            if key == "no_of_points":
                continue

            if hasattr(self, key) and not key.startswith('_'):
                setattr(self, key, value)
            else:
                print(f"Warning: Setting '{key}' does not exist.")

            if (key == 'no_of_anchors' or
                key == 'no_of_tags' or
                key == 'n_components' or
                key == 'random_anchors'):
                if self.is_real_data is False:
                    self.set_random_tags()

    def set_random_tags(self):
        _ANCHORS = np.array([
                [self.x_min, self.y_min, self.z_min],  # Bottom origin
                [self.x_max, self.y_min, self.z_min],
                [self.x_max, self.y_max, self.z_min],
                [self.x_min, self.y_max, self.z_min],
                [self.x_min, self.y_min, self.z_max],  # Top origin
                [self.x_max, self.y_min, self.z_max],
                [self.x_max, self.y_max, self.z_max],
                [self.x_min, self.y_max, self.z_max],
            [(self.x_max - self.x_min) / 2, (self.y_max - self.y_min) / 2, (self.z_max - self.z_min) / 2],
            [(self.x_max - self.x_min) / 4, (self.y_max - self.y_min) / 4, (self.z_max - self.z_min) / 4],
            [3 * (self.x_max - self.x_min) / 4, (self.y_max - self.y_min) / 4, (self.z_max - self.z_min) / 4],
            [(self.x_max - self.x_min) / 4, 3 * (self.y_max - self.y_min) / 4, (self.z_max - self.z_min) / 4],
            [3 * (self.x_max - self.x_min) / 4, 3 * (self.y_max - self.y_min) / 4, (self.z_max - self.z_min) / 4],
        ])

        low_bounds = [self.x_min, self.y_min, self.z_min][:self.n_components]
        high_bounds = [self.x_max, self.y_max, self.z_max][:self.n_components]

        self.tags = np.random.uniform(
            low=low_bounds,
            high=high_bounds,
            size=(self.no_of_tags, self.n_components)
        )

        if getattr(self, 'random_anchors', False):
            self.anchors = np.random.uniform(
                low=low_bounds,
                high=high_bounds,
                size=(self.no_of_anchors, self.n_components)
            )
        else:
            actual_anchors = min(self.no_of_anchors, len(_ANCHORS))
            self.anchors = _ANCHORS[:actual_anchors, :self.n_components]

        self.points = np.vstack((self.anchors, self.tags))
        self.true_distance = euclidean_distances(self.points)
        self.triang_dist = self.true_distance[self.no_of_anchors, :self.no_of_anchors]

    def process_real_data(self):
        selected_anchors = self.select_anchors()
        selected_tags = [i for i in range(self.no_of_points) if i not in selected_anchors]
        self.real_data = self.real_data[list(selected_anchors) + list(selected_tags)]
        self.anchors = np.array([point['real_coordinates'][:self.n_components] for point in self.real_data[:self.no_of_anchors]])
        self.tags = np.array([point['real_coordinates'][:self.n_components] for point in self.real_data[self.no_of_anchors:]])
        self.points = np.vstack((self.anchors, self.tags))
        self.true_distance = euclidean_distances(self.points)

        distances, _ = compute_mds_wifi_similarity(self.real_data, cosine, 'All')

        physical_max_distance = np.max(self.true_distance)
        print(f'physical_max_distance: {physical_max_distance}')
        mask = distances < 0.99
        if np.any(mask):
            wifi_max_distance = np.max(distances[mask])
            scale_factor = physical_max_distance / wifi_max_distance
            distances = distances * scale_factor

        distances[:self.no_of_anchors, :self.no_of_anchors] = euclidean_distances(self.anchors)
        self.distance = distances


    def select_anchors(self):
        if self.is_random:
            anchors_index = np.random.choice(self.no_of_points, size=self.no_of_anchors, replace=False)
        else:
            anchors_index = np.linspace(0, self.no_of_points - 1, self.no_of_anchors, dtype=int)
        print(f"Am ales acești indici pentru ancore: {anchors_index}")

        return anchors_index



