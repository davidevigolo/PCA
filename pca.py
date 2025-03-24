import numpy as np

def build_cov_mat(data):
    return np.cov(data, rowvar=False)

def spectral_decomposition(matrix):
    eigvals, eigvecs = np.linalg.eig(matrix)
    return eigvecs.T, eigvals  # Rows are eigenvectors

def pca(data):
    data = np.array(data, dtype=float)
    
    # 1. Standardize FIRST
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0, ddof=1)  # Use sample std (ddof=1)
    data_std = (data - mean) / std
    
    # 2. Compute covariance of standardized data
    cov_matrix = build_cov_mat(data_std)
    
    # 3. Spectral decomposition
    eigvecs, eigvals = spectral_decomposition(cov_matrix)
    
    # 4. Sort eigenvectors by eigenvalues (descending)
    sorted_indices = np.argsort(eigvals)[::-1]
    eigvecs = eigvecs[sorted_indices, :]  # Sort rows
    eigvals = eigvals[sorted_indices]
    
    # 5. Select top 2 eigenvectors (rows)
    top_eigvecs = eigvecs[:2, :]
    
    # 6. Project standardized data
    projected_data = (top_eigvecs @ data_std.T).T
    
    return projected_data