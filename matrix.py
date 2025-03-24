import numpy as np

import matplotlib.pyplot as plt

def read_data(file_name):
    with open(file_name, 'r') as f:
        data = f.readlines()
    data = [line.strip().split(',') for line in data]
    return data

def stampa_grafici_dispersione(x, y):
    plt.scatter(x, y)
    plt.title("Scatter Plot")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.show()