import numpy as np

# Data una matrice costruisce la matrice di covarianza
def build_cov_mat(data):
    data = np.array(data, dtype=float)
    cov_matrix = np.cov(data, rowvar=False)
    return cov_matrix

# restituisce la matrice trasposta degli autovettori U^t della decomposizione spettrale di una data matrice
def spectral_decomposition(matrix):
    eigvals, eigvecs = np.linalg.eig(matrix)
    return eigvecs.T,eigvals

# Data una matrice di dati restituisce la matrice dei dati trasformata in base alle prime due componenti principali
def pca(data):
    cov_matrix = build_cov_mat(data)
    eigvecs, eigvals = spectral_decomposition(cov_matrix)

    # ordina gli autovettori in ordine decrescente
    sorted_indices = np.argsort(eigvals)[::-1]
    eigvecs = eigvecs[:, sorted_indices]
    eigvals = eigvals[sorted_indices]

    data_matrix = np.array(data)
    # seleziona le prime due componenti principali, sono ordinati in ordine decrescente quindi i primi due sono i maggiori
    eigvecs = eigvecs[:, :2]

    # standardizzazione dei dati
    standard_deviation = np.std(data, axis=0)
    data_mean = data_matrix - np.mean(data, axis=0)
    data_matrix_std = data_mean / standard_deviation

    # proiezione dei dati sulla base degli autovettori scelti
    rescaled_data = (eigvecs.T @ data_matrix_std.T)

    return rescaled_data.T
    