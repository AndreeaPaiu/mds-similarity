import numpy as np
from matplotlib import pyplot as plt
import os
import matplotlib.lines as mlines

from smacofAnchors.MDS_A.test.utils import rmse

COLORS = iter(['blue', 'red', 'green', 'magenta'])

def plot_rmse(first_quartile, median, third_quartile, x_axis=None):
	plt.figure(1)
	if x_axis is None:
		x_axis = np.arange(median.shape[0])
	color = next(COLORS)
	plot1, = plt.plot(x_axis, median, '-', color=color)
	plot2 = plt.fill_between(x_axis, first_quartile, third_quartile, color=color, alpha=0.3)
	axes = plt.gca()
	return plot1, plot2, axes

def plot_rmse_vs_noise(*args, **kwargs):
	plot1, plot2, axes = plot_rmse(*args, **kwargs)
	axes.set_xlabel(r'Measurement noise $\sigma$ [m]')
	axes.set_ylabel('RMSE of computed configuration [m]')
	return plot1, plot2

def plot_rmse_vs_anchors(*args, **kwargs):
	plot1, plot2, axes = plot_rmse(*args, **kwargs)
	axes.set_xlabel('No of anchors')
	axes.set_ylabel('RMSE of computed configuration [m]')
	return plot1, plot2

def plot_rmse_vs_ntags(*args, **kwargs):
	plot1, plot2, axes = plot_rmse(*args, **kwargs)
	axes.set_xlabel('No of tags')
	axes.set_ylabel('RMSE of computed configuration [m]')
	return plot1, plot2


def plot_3d_real_vs_estimated(config, results_embedding):
	"""
    Generează un plot 3D interactiv comparând pozițiile reale cu cele estimate.
    Real Anchors (Triunghiuri Roșii)
    Real Tags (Puncte Verzi)
    Estimated Tags (X-uri Albastre)
    """
	fig = plt.figure(figsize=(12, 9))
	ax = fig.add_subplot(111, projection='3d')

	# --- 1. Separarea Datelor din Config și Rezultate ---
	# Deoarece sunt rearanjate, ancorele sunt primele no_of_anchors elemente

	# Datele REALE din Config
	real_anchors = config.points[:config.no_of_anchors]
	real_tags = config.points[config.no_of_anchors:]

	# Datele ESTIMATE de MDS
	# Deoarece ancorele sunt fixe, estimările pentru ele ar trebui să fie egale cu cele reale
	estimated_tags = results_embedding[config.no_of_anchors:]

	# --- 2. Plotarea Punctelor Reale (Bazele) ---
	# Ancorele Reale (Triunghiuri Roșii Mari)
	ax.scatter(real_anchors[:, 0], real_anchors[:, 1],
			   color='red', marker='^', s=150, label='Real Anchors (Fixed)')

	# Tag-urile Reale (Puncte Verzi)
	ax.scatter(real_tags[:, 0], real_tags[:, 1],
			   color='green', marker='o', s=50, label='Real Position')

	# --- 3. Plotarea Punctelor Estimate ---
	# Tag-urile Estimate (X-uri Albastre)
	ax.scatter(estimated_tags[:, 0], estimated_tags[:, 1],
			   color='blue', marker='x', s=50, label='MDS Estimation')

	# --- 4. Desenarea Liniilor de Eroare ---
	# Tragem o linie punctată gri între poziția reală și cea estimată a fiecărui tag
	for i in range(len(real_tags)):
		ax.plot([real_tags[i, 0], estimated_tags[i, 0]],
				[real_tags[i, 1], estimated_tags[i, 1]],
				# [real_tags[i, 2], estimated_tags[i, 2]],
				color='gray', linestyle='--', alpha=0.5)

	# --- 5. Titlu și Etichete ---
	# Calculăm eroarea doar pe tag-uri (fără ancore) pentru titlu
	eroare_tags = rmse(estimated_tags, real_tags)

	ax.set_xlabel('Axa X (metri)')
	ax.set_ylabel('Axa Y (metri)')
	ax.set_zlabel('Axa Z (metri)')
	ax.set_title(f"Vizualizare Rezultate Real-WiFi | RMSE pe Tags: {eroare_tags:.2f} m")
	ax.legend()

	# --- 6. Setarea View-ului (Opțional) ---
	# ax.view_init(elev=30, azim=45) # Poți schimba unghiul implicit de start

	print("Graficul s-a randat! Poți roti imaginea cu mouse-ul.")
	plt.show()
	plt.close('all')

def plot_rmse_vs_missing_ratio(*args, **kwargs):
	plot1, plot2, axes = plot_rmse(*args, **kwargs)
	axes.set_xlabel('Percentage of missing tag-to-tag distances (%)')
	axes.set_ylabel('RMSE of computed configuration [m]')
	return plot1, plot2


def save_real_vs_estimated_plot(config, results_embedding, ratio):
	fig = plt.figure(figsize=(12, 10))  # Am mărit puțin figura pentru claritate

	is_3d = config.n_components == 3
	if is_3d:
		ax = fig.add_subplot(111, projection='3d')
	else:
		ax = fig.add_subplot(111)

	real_anchors = config.points[:config.no_of_anchors]
	real_tags = config.points[config.no_of_anchors:]
	estimated_tags = results_embedding[config.no_of_anchors:]

	# ---- 1. Desenăm topologia rețelei (conexiunile tag-to-tag) ----
	if hasattr(config, 'weight_matrix') and config.weight_matrix is not None:
		n_anchors = config.no_of_anchors
		n_points = config.no_of_points

		# Parcurgem doar blocul tag-urilor din matricea de ponderi
		for i in range(n_anchors, n_points):
			for j in range(i + 1, n_points):
				p1 = config.points[i]
				p2 = config.points[j]

				# Dacă greutatea e 0, distanța lipsește
				if config.weight_matrix[i, j] == 0:
					if is_3d:
						ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], color='red', linestyle=':', alpha=0.3)
					else:
						ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color='red', linestyle=':', alpha=0.3)
				else:
					# Dacă greutatea e 1, distanța este activă
					if is_3d:
						ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], color='lightgreen', linestyle='-',
								alpha=0.15)
					else:
						ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color='lightgreen', linestyle='-', alpha=0.15)

	# ---- 2. Desenăm punctele (Ancore, Real, Estimat) ----
	# Am folosit zorder=5 ca să ne asigurăm că punctele sunt desenate PESTE liniile rețelei
	if is_3d:
		ax.scatter(real_anchors[:, 0], real_anchors[:, 1], real_anchors[:, 2], c='red', marker='^', s=150,
				   label='Anchors', zorder=5)
		ax.scatter(real_tags[:, 0], real_tags[:, 1], real_tags[:, 2], c='green', marker='o', s=50,
				   label='Tags (Real)', zorder=5)
		ax.scatter(estimated_tags[:, 0], estimated_tags[:, 1], estimated_tags[:, 2], c='blue', marker='x', s=50,
				   label='Tags (Estimation)', zorder=5)

		# Liniile de eroare (Real -> Estimat) desenate cu negru
		for i in range(len(real_tags)):
			ax.plot([real_tags[i, 0], estimated_tags[i, 0]],
					[real_tags[i, 1], estimated_tags[i, 1]],
					[real_tags[i, 2], estimated_tags[i, 2]], color='black', linestyle='--', alpha=0.5)
	else:
		ax.scatter(real_anchors[:, 0], real_anchors[:, 1], c='red', marker='^', s=150, label='Ancore', zorder=5)
		ax.scatter(real_tags[:, 0], real_tags[:, 1], c='green', marker='o', s=50, label='Tag-uri (Real)', zorder=5)
		ax.scatter(estimated_tags[:, 0], estimated_tags[:, 1], c='blue', marker='x', s=50, label='Tag-uri (Estimat)',
				   zorder=5)

		for i in range(len(real_tags)):
			ax.plot([real_tags[i, 0], estimated_tags[i, 0]],
					[real_tags[i, 1], estimated_tags[i, 1]], color='black', linestyle='--', alpha=0.5)

	# ---- 3. Customizarea Legendei și a Titlului ----
	eroare_tags = rmse(estimated_tags, real_tags)
	procent = int(ratio * 100)
	ax.set_title(f"MDS-A | Missing connections: {procent}% | RMSE: {eroare_tags:.2f} m")

	# Construim legenda incluzând și liniile create de noi
	handles, labels = ax.get_legend_handles_labels()
	linie_activa = mlines.Line2D([], [], color='lightgreen', linestyle='-', label='Active connection')
	linie_lipsa = mlines.Line2D([], [], color='red', linestyle=':', label='Missing connection')
	linie_eroare = mlines.Line2D([], [], color='black', linestyle='--', label='Estimation error')
	handles.extend([linie_activa, linie_lipsa, linie_eroare])

	# Punem legenda în exterior dreapta pentru a nu acoperi graficul
	ax.legend(handles=handles, loc='center left', bbox_to_anchor=(1, 0.5))

	# ---- 4. Salvarea imaginii ----
	folder_export = 'rezultate_ploturi'
	if not os.path.exists(folder_export):
		os.makedirs(folder_export)

	filename = f'{folder_export}/estimare_lipsa_{procent}pct.png'
	# Folosim bbox_inches='tight' pentru ca legenda scoasă în exterior să fie cuprinsă în poză
	plt.savefig(filename, dpi=150, bbox_inches='tight')
	plt.close(fig)