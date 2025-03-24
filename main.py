from matrix import *
from regressione import *
from pca import *

def main():
    # Lettura dei dati
    data = read_data('./dati/dati.csv')
    tmin = [float(row[1]) for row in data]
    tmed = [float(row[2]) for row in data]
    tmax = [float(row[3]) for row in data]
    ptot = [float(row[4]) for row in data]
    data = list(zip(tmin, tmed, tmax, ptot))

    a,b = regressione(tmin, tmed) # coefficienti della retta di regressione a*x + b

    # primo campione bivariato tmin,tmed
    plt.figure(1)
    plt.plot(tmin, tmed, 'o', label='Dati')
    plt.plot(tmin, [a * xi + b for xi in tmin], 'r', label='Retta di regressione') # per ogni punto xi nel dataset, valuta la retta di regressione in quel punto
    plt.ylabel('Tmed')
    plt.xlabel('Tmin')
    plt.title('Regressione lineare sul primo campione bivariato (tmin,tmed)')
    plt.legend()
    plt.show()

    # secondo campione bivariato tmin,ptot
    a2, b2 = regressione(tmin, ptot)
    print(f"Coefficients for the second regression line: a = {a2}, b = {b2}")

    plt.figure(2)
    plt.plot(tmin, ptot, 'o', label='Dati')
    plt.plot(tmin, [a2 * xi + b2 for xi in tmin], 'r', label='Retta di regressione') # per ogni punto xi nel dataset, valuta la retta di regressione in quel punto
    plt.xlabel('Tmin')
    plt.ylabel('Ptot')
    plt.title('Regressione lineare sul secondo campione bivariato (tmin,ptot)')
    plt.legend()
    plt.show()

    shrinked_data = pca(data)
    # Plot dei dati proiettati sulle prime due componenti principali
    plt.figure(3)
    plt.scatter(shrinked_data[:, 0], shrinked_data[:, 1])
    plt.xlabel('Componente 1')
    plt.ylabel('Componente 2')
    plt.title('PCA')
    plt.legend()
    plt.show()


if __name__ == "__main__":
        main()