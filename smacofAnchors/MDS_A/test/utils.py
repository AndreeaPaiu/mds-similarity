import numpy as np

from smacofAnchors.MDS_A.test import mds


def rmse(computed, real):
    return np.sqrt(((computed - real) ** 2).mean())


def get_triangulation_initial_guess(config):
    """Calculează estimarea geometrică folosind o marjă de eroare adaptivă."""
    rezultat_final = list(config.anchors.copy())
    centru = np.mean(config.anchors, axis=0)

    # Definim treptele de relaxare a erorii (cât de mult 'umflăm' inelele)
    # Începe cu 0.1, apoi 0.5, 1.0, 1.5, ... până la un maxim de 5.0 metri extra
    trepte_eroare = [0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]

    for id, tag in enumerate(config.tags):
        distante = config.distance[config.no_of_anchors + id, :config.no_of_anchors]

        estimare = None
        # Căutăm o intersecție mărind progresiv marja de eroare
        for extra_error in trepte_eroare:
            estimare = estimate_position(config.anchors, distante, error=config.sigma + extra_error)
            if estimare is not None:
                # Am găsit intersecția! Oprim căutarea pentru acest tag.
                break

        if estimare is not None:
            # Protecție 3D vs 2D
            if config.n_components == 3 and len(estimare) == 2:
                estimare = np.append(estimare, 0.0)
            rezultat_final.append(estimare)
        else:
            # Dacă inelele nu s-au atins nici măcar cu +5.0m eroare,
            # datele sunt complet compromise. Folosim fallback-ul răsfirat.
            scatter = np.random.uniform(-3, 3, size=config.n_components)
            punct_rasfirat = centru + scatter

            if config.n_components == 3 and punct_rasfirat[2] < 0:
                punct_rasfirat[2] = 0.1

            rezultat_final.append(punct_rasfirat)

    return np.array(rezultat_final)

def first_third_quartile_and_median(data):
    first_quartile = np.percentile(data, 25, axis=1)
    third_quartile = np.percentile(data, 75, axis=1)
    median = np.median(data, axis=1)
    return first_quartile, median, third_quartile

def estimate_coords(config):
    mds_ = mds.MDS(config, config.algorithm, n_components=config.n_components, max_iter=config.max_iter, eps=config.eps,
                   random_state=config.random_state, n_jobs=config.n_jobs,
                   dissimilarity="precomputed", metric=True, verbose=0, n_init=config.n_init)

    results = mds_.fit(config.distance, init=config.initial_guess)
    return results


def add_noise(config):
    # Pornește de la distanțele reale (fără zgomot)
    distance = config.true_distance.copy()
    n_points = config.no_of_points
    n_tags = config.no_of_tags
    n_anchors = config.no_of_anchors

    # 1. Matricea de greutăți (toate conexiunile = 1 inițial)
    weights = np.ones((n_points, n_points))
    np.fill_diagonal(weights, 0)

    # EXCLUDEM ANCORELE: Distanțele dintre ele sunt perfecte și nu se optimizează
    weights[:n_anchors, :n_anchors] = 1

    # 2. Ștergem aleatoriu procentul dorit din conexiunile Tag-to-Tag
    missing_ratio = getattr(config, 'missing_data_ratio', 0.0)
    if getattr(config, 'missingData', False):
        missing_ratio = 1.0

    if missing_ratio > 0:
        n_pairs = n_tags * (n_tags - 1) // 2
        missing_count = int(n_pairs * missing_ratio)

        tag_weights = np.ones((n_tags, n_tags))
        np.fill_diagonal(tag_weights, 0)

        if missing_count > 0:
            upper_indices = np.triu_indices(n_tags, k=1)
            random_choices = np.random.choice(n_pairs, missing_count, replace=False)

            for idx in random_choices:
                i = upper_indices[0][idx]
                j = upper_indices[1][idx]
                tag_weights[i, j] = 0
                tag_weights[j, i] = 0

        weights[-n_tags:, -n_tags:] = tag_weights

    # SALVĂM matricea ca să o poată citi funcția de plotare și algoritmul MDS
    config.weight_matrix = weights

    # 3. Adăugăm zgomot STRICT pe conexiunile Anchor-Tag și Tag-Tag
    if config.sigma > 0:
        # Generăm zgomot doar pentru blocul ancore-tag
        noise_at = np.random.normal(config.mu, config.sigma, size=(n_anchors, n_tags))
        distance[:n_anchors, -n_tags:] += noise_at
        distance[-n_tags:, :n_anchors] += noise_at.T

        # Generăm zgomot pentru blocul tag-tag (simetric)
        noise_tt = np.random.normal(config.mu, config.sigma, size=(n_tags, n_tags))
        noise_tt = (noise_tt + noise_tt.T) / 2.0
        distance[-n_tags:, -n_tags:] += noise_tt

    # 5. PROTECȚIE ANTI-NEGATIVE
    # Acolo unde există o conexiune (weight > 0) și zgomotul a făcut distanța prea mică, o limităm la 0.1
    distance = np.where((weights > 0) & (distance < 0.1), 0.1, distance)
    np.fill_diagonal(distance, 0)

    # Restaurăm distanțele perfecte între ancore
    distance[:n_anchors, :n_anchors] = config.true_distance[:n_anchors, :n_anchors]

    config.distance = distance


def compute_estimations(config):
    if config.is_real_data is False:
        add_noise(config)

    if getattr(config, 'use_triangulation_init', False):
        config.initial_guess = get_triangulation_initial_guess(config)
    results = estimate_coords(config)
    coords = results.embedding_

    return coords

def _get_circle_intersections(c1, r1, c2, r2):
    """Calculează punctele de intersecție dintre două cercuri."""
    d = np.linalg.norm(c1 - c2)
    # Verificăm dacă cercurile sunt prea departe, unul e în altul, sau sunt concentrice
    if d > r1 + r2 or d < abs(r1 - r2) or d == 0:
        return []

    a = (r1 ** 2 - r2 ** 2 + d ** 2) / (2 * d)
    h = np.sqrt(max(0, r1 ** 2 - a ** 2))
    p2 = c1 + a * (c2 - c1) / d
    x_inv = (h / d) * (c2[1] - c1[1])
    y_inv = (h / d) * (c2[0] - c1[0])

    return [np.array([p2[0] + x_inv, p2[1] - y_inv]), np.array([p2[0] - x_inv, p2[1] + y_inv])]


def estimate_position(anchors, distances, error=0.5):
    """
    Estimează poziția unui punct folosind trilaterație cu marjă de eroare geometrică.

    :param anchors: Listă sau array cu coordonatele ancorelor (ex: [[0,0], [8,0], [0,6]])
    :param distances: Listă cu distanțele măsurate până la fiecare ancoră (ex: [10.0, 6.0, 8.0])
    :param error: Marja de eroare a senzorilor (+/- metri)
    :return: Array numpy [X, Y] cu poziția estimată, sau None dacă nu există o zonă validă.
    """
    if len(anchors) != len(distances):
        raise ValueError("Numărul ancorelor trebuie să fie egal cu numărul distanțelor!")

    centers = [np.array(p) for p in anchors]

    # 1. Generăm limitele inelelor de incertitudine pentru fiecare ancoră
    circles = []
    for i in range(len(centers)):
        # Raza minimă nu poate fi negativă
        r_min = max(0, distances[i] - error)
        r_max = distances[i] + error
        circles.append({'c': centers[i], 'r': r_min})
        circles.append({'c': centers[i], 'r': r_max})

    # 2. Găsim toate intersecțiile posibile între aceste granițe
    all_pts = []
    for i in range(len(circles)):
        for j in range(i + 1, len(circles)):
            all_pts.extend(_get_circle_intersections(circles[i]['c'], circles[i]['r'],
                                                     circles[j]['c'], circles[j]['r']))

    # 3. Păstrăm DOAR punctele care respectă simultan distanța față de TOATE ancorele
    valid_vertices = []
    for pt in all_pts:
        is_valid = True
        for i in range(len(centers)):
            d_to_anchor = np.linalg.norm(pt - centers[i])
            # 1e-7 este o toleranță microscopică pentru erori de rotunjire specifice float-urilor
            if not ((distances[i] - error - 1e-7) <= d_to_anchor <= (distances[i] + error + 1e-7)):
                is_valid = False
                break
        if is_valid:
            valid_vertices.append(pt)

    # 4. Curățăm rezultatele (eliminăm intersecțiile care se suprapun)
    unique_vertices = []
    for v in valid_vertices:
        if not any(np.linalg.norm(v - u) < 1e-6 for u in unique_vertices):
            unique_vertices.append(v)

    # 5. Returnăm centroidul (media coordonatelor zonei de intersecție)
    if unique_vertices:
        return np.mean(unique_vertices, axis=0)
    else:
        return None

def compute_triangulation(config):
    add_noise(config)

    coords = [
        estimate_position(
            config.anchors,
            config.distance[config.no_of_anchors + id, :config.no_of_anchors],
            config.sigma
        )
        for id, tag in enumerate(config.tags)
    ]

    return coords
