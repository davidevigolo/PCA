import statistics

"""
    dato un campione bi-variato in x, y calcola:
        - a* -> il coefficiente angolare della retta di regressione
        - b* -> l'intercetta di tale retta
        
    nota: usiamo statistics.mean() per una maggiore stabilità
          numerica nel risultato, verificato tramite vari test
"""
def regressione(x, y):
    # calcolo media dei valori in x e y
    media_x = statistics.mean(x)
    media_y = statistics.mean(y)
    
    # calcolo numeratore e denominatore tramite formula precedentemente ricavata
    num = sum((xi - media_x) * (yi - media_y) for xi, yi in zip(x, y))
    den = sum((xi - media_x) ** 2 for xi in x)
    
    # calcolo di a* e b*
    a_s = num / den
    b_s = media_y - a_s * media_x
    return a_s, b_s
