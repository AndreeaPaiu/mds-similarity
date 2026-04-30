import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import cosine
from sklearn.metrics import euclidean_distances

# Importurile tale
from smacofAnchors.MDS_A.test import mds
from smacofAnchors.MDS_A.test.config import SettingsManager
from smacofAnchors.MDS_A.test.utils import rmse
from test_mds.compute_mds_wifi_similarity import compute_mds_wifi_similarity
from test_mds.preprocessing_file import *


def pregateste_date_reale(config, date_wifi, D, indici_ancore):
    """Rearanjează datele, izolează distanțele lipsă și scalează corect."""
    from sklearn.metrics import euclidean_distances

    coordonate_reale = np.array([punct['real_coordinates'] for punct in date_wifi])
    N = len(coordonate_reale)

    indici_taguri = [i for i in range(N) if i not in indici_ancore]
    noua_ordine = indici_ancore + indici_taguri

    puncte_rearanjate = coordonate_reale[noua_ordine]
    D_rearanjata = D[noua_ordine, :][:, noua_ordine]

    # --- 1. REZOLVĂM DATELE LIPSĂ (Weight Matrix) ---
    # Găsim unde matricea are valoarea 1 (cu o mică toleranță pt floats)
    masca_lipsa = np.isclose(D_rearanjata, 1.0)

    # Creăm matricea de greutăți (1 = folosim distanța, 0 = o ignorăm)
    weights = np.ones((N, N))
    weights[masca_lipsa] = 0

    # --- 2. SCALAREA CORECTĂ (Fără a lua în calcul datele lipsă) ---
    distante_fizice = euclidean_distances(puncte_rearanjate)
    distanta_maxima_fizica = np.max(distante_fizice)

    # Căutăm maximul Wi-Fi DOAR unde NU avem date lipsă (unde masca e False)
    if np.any(~masca_lipsa):
        distanta_maxima_wifi = np.max(D_rearanjata[~masca_lipsa])
    else:
        distanta_maxima_wifi = 1.0

    if distanta_maxima_wifi > 0:
        factor_scalare = distanta_maxima_fizica / distanta_maxima_wifi
        D_scalata = D_rearanjata * factor_scalare
    else:
        D_scalata = D_rearanjata

    # --- 3. ACTUALIZARE CONFIG ---
    config.no_of_anchors = len(indici_ancore)
    config.no_of_tags = len(indici_taguri)
    config.n_components = 3
    config.anchors = puncte_rearanjate[:config.no_of_anchors]
    config.tags = puncte_rearanjate[config.no_of_anchors:]
    config.points = puncte_rearanjate

    config.distance = D_scalata

    # INJECTĂM MATRICEA DE GREUTĂȚI ÎN CONFIG PENTRU A FI FOLOSITĂ DE MDS
    config.weight_matrix = weights

def plot_3d_comparison(config, coord_estimate):
    """Desenează un grafic 3D clar cu punctele reale vs cele estimate."""
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')

    # Separăm datele
    ancore = coord_estimate[:config.no_of_anchors]
    tags_reale = config.points[config.no_of_anchors:]
    tags_estimate = coord_estimate[config.no_of_anchors:]

    # Desenăm Ancorele (Triunghiuri Roșii Mari)
    ax.scatter(ancore[:, 0], ancore[:, 1], ancore[:, 2],
               c='red', marker='^', s=150, label='Ancore (Fixe)')

    # Desenăm Tag-urile Reale (Puncte Verzi)
    ax.scatter(tags_reale[:, 0], tags_reale[:, 1], tags_reale[:, 2],
               c='green', marker='o', s=50, label='Poziție Reală')

    # Desenăm Tag-urile Estimate (X-uri Albastre)
    ax.scatter(tags_estimate[:, 0], tags_estimate[:, 1], tags_estimate[:, 2],
               c='blue', marker='x', s=50, label='Estimare MDS')

    # Tragem o linie între poziția reală și cea estimată ca să vedem "eroarea" vizual
    for i in range(len(tags_reale)):
        ax.plot([tags_reale[i, 0], tags_estimate[i, 0]],
                [tags_reale[i, 1], tags_estimate[i, 1]],
                [tags_reale[i, 2], tags_estimate[i, 2]],
                color='gray', linestyle='--', alpha=0.5)

    # Calculăm eroarea doar pentru titlu
    eroare_medie = rmse(tags_estimate, tags_reale)

    ax.set_xlabel('X (metri)')
    ax.set_ylabel('Y (metri)')
    ax.set_zlabel('Z (metri)')
    ax.set_title(f"Vizualizare MDS-A cu 4 Ancore | Eroare Medie (RMSE): {eroare_medie:.2f} m")
    ax.legend()

    # Afișăm graficul interactiv
    plt.show()


def test_mds_ancore_multiple(config, date_wifi, matrice_D_reala):
    print("Pregătim datele pentru ancorele multiple...")

    # =======================================================
    # AICI ADAUGI ANCORELE TALE!
    # Alege indici din datele tale.
    # Recomandare pentru o clădire 3D (Minim 8 ancore):
    # - 4 indici de la etajul cel mai de jos (în cele 4 colțuri)
    # - 4 indici de la etajul cel mai de sus (în cele 4 colțuri)
    # =======================================================
    indici_ancore = [
        0, 10, 20, 30,

        # 4 puncte de la etajul de sus (Z mare) -> Alege indicii corecți din datele tale!
        45, 55, 65, 75
    ]

    # Asigură-te că nu ai indici dublați sau în afara listei
    indici_ancore = list(set(indici_ancore))
    indici_ancore = [i for i in indici_ancore if i < len(date_wifi)]

    print(f"Am selectat {len(indici_ancore)} ancore.")

    pregateste_date_reale(config, date_wifi, matrice_D_reala, indici_ancore)

    print("Rulăm algoritmul MDS...")
    mds_ = mds.MDS(
        config,
        algorithm='_smacof_with_anchors',
        n_components=3,
        max_iter=500,
        eps=1e-14,
        random_state=np.random.RandomState(seed=42),
        n_jobs=1,
        dissimilarity="precomputed",
        metric=True,
        verbose=0,
        n_init=10  # Previne colapsul în minime locale
    )

    results = mds_.fit(config.distance)

    # Calculăm eroarea doar pe tag-urile estimate (fără ancore)
    tags_estimate = results.embedding_[config.no_of_anchors:]
    tags_reale = config.points[config.no_of_anchors:]
    eroare_medie = rmse(tags_estimate, tags_reale)

    print(f"Calcul finalizat! Eroare Medie (RMSE): {eroare_medie:.2f} metri.")
    print("Generăm graficul 3D...")

    plot_3d_comparison(config, results.embedding_)

def test_simplu_4_ancore(config, date_wifi, matrice_D_reala):
    print("Pregătim datele pentru cele 4 ancore...")

    # AICI ALEGI ANCORELE: Pune indicii a 4 puncte care vrei să fie ancore.
    # Recomand să alegi 4 puncte care știi că sunt cât de cât pe marginile camerei/holului
    indici_ancore = [0, 10, 20, 30]

    pregateste_date_reale(config, date_wifi, matrice_D_reala, indici_ancore)

    print("Rulăm algoritmul MDS...")
    mds_ = mds.MDS(
        config,
        algorithm='_smacof_with_anchors',
        n_components=3,
        max_iter=500,
        eps=1e-14,
        random_state=np.random.RandomState(seed=42),
        n_jobs=1,
        dissimilarity="precomputed",
        metric=True,
        verbose=0,
        n_init=10  # Mai multe încercări pentru a găsi cea mai bună pliere
    )

    results = mds_.fit(config.distance)

    print("Calcul finalizat! Generăm graficul 3D...")
    plot_3d_comparison(config, results.embedding_)


if __name__ == "__main__":
    config = SettingsManager()

    absolut_path = sys.argv[1]
    files_names = sys.argv[2:]

    # Încărcarea datelor tale reale
    preprocessing_files_data = []
    merged_data = []

    for file_name in files_names[:2]:
        file_data = preprocessing_required_data(absolut_path, file_name, file_name[0])
        preprocessing_files_data.append(file_data)
        merged_data += file_data

    similarities, _ = compute_mds_wifi_similarity(merged_data, cosine, 'All')
    date_wifi = merged_data
    matrice_D_reala = similarities

    # Rulăm testul nostru simplu, vizual
    test_simplu_4_ancore(config, date_wifi, matrice_D_reala)
    # test_mds_ancore_multiple(config, date_wifi, matrice_D_reala)