import numpy as np

def min_max_normalize(x, axis=None):
    min = np.min(x, axis=axis)
    max = np.max(x, axis=axis)
    return (x-min)/(max-min)

def z_normalize(x, axis=None):
    mean = np.mean(x, axis=axis)
    std = np.std(x, axis=axis)
    return (x - mean)/std
