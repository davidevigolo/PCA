from matrix import *
from regressione import *
from pca import *

def main():
    data = read_data('./dati/dati.csv')
    tmin = [float(row[1]) for row in data]
    tmed = [float(row[2]) for row in data]
    tmax = [float(row[3]) for row in data]
    ptot = [float(row[4]) for row in data]
    data = list(zip(tmin, tmed, tmax, ptot))
    
    shrinked_data = pca(data)
    # Plot the rescaled data for the two components with the largest eigenvalues
    plt.figure(3)
    plt.scatter(shrinked_data[:, 0], shrinked_data[:, 1], label='Rescaled Data')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.legend()
    plt.show()

    a,b = regressione(tmin, tmed)
    print(f"Coefficients for the first regression line: a = {a}, b = {b}")

    # primo campione bivariato tmin,tmed
    plt.figure(1)
    plt.plot(tmin, tmed, 'o', label='Dati')
    plt.plot(tmin, [a * xi + b for xi in tmin], 'r', label='Retta di regressione') # per ogni punto xi nel dataset, valuta la retta di regressione in quel punto
    plt.ylabel('Tmed')
    plt.xlabel('Tmin')
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
    plt.legend()
    plt.show()


if __name__ == "__main__":
        main()