import statistics

"""
    dato un campione bi-variato in x, y calcola:
        - a* -> il coefficiente angolare della retta di regressione
        - b* -> l'intercetta di tale retta
        
    nota: usiamo statistics.mean() per una maggiore stabilita
          numerica nel risultato, verificato tramite vari test
"""
def regression(x, y):
    # calcolo media dei valori in x e y
    mean_x = statistics.mean(x)
    mean_y = statistics.mean(y)
    
    # calcolo numeratore e denominatore tramite
    # formula precedentemente ricavata
    num = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
    den = sum((xi - mean_x) ** 2 for xi in x)
    
    # calcolo di a* e b*
    a_s = num / den
    b_s = mean_y - a_s * mean_x
    return a_s, b_s
