import numpy as np

def costruisci_mat_covarianza(data):
    return np.cov(data, rowvar=False)

def decomposizione_spettrale(matrice):
    autovalori, autovettori = np.linalg.eig(matrice)
    return autovettori.T, autovalori  # Le righe sono autovettori

def pca(data):
    data = np.array(data, dtype=float)
    
    # 1. Standardizzazione dei dati
    media = np.mean(data, axis=0)
    std = np.std(data, axis=0, ddof=1)  # Usa la deviazione standard del campione (ddof=1)
    data_std = (data - media) / std
    
    # 2. Calcola la matrice di covarianza dei dati standardizzati
    matrice_covarianza = costruisci_mat_covarianza(data_std)
    
    # 3. Decomposizione spettrale
    autovettori, autovalori = decomposizione_spettrale(matrice_covarianza)
    
    # 4. Ordina gli autovettori per autovalori (decrescente)
    indici_ordinati = np.argsort(autovalori)[::-1]
    autovettori = autovettori[indici_ordinati, :]  # Ordina le righe
    autovalori = autovalori[indici_ordinati]
    
    # 5. Seleziona i primi 2 autovettori (righe)
    max_autovettori = autovettori[:2, :]
    
    # 6. Proietta i dati standardizzati
    dati_proiettati = (max_autovettori @ data_std.T).T
    
    return dati_proiettati