import numpy as np

import matplotlib.pyplot as plt

def read_data(file_name):
    with open(file_name, 'r') as f:
        data = f.readlines()
    data = [line.strip().split(',') for line in data]
    return data