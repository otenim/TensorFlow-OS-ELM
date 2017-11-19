import numpy as np
import os
import pickle as pkl

def min_max_normalize(x, axis=None):
    min = np.min(x, axis=axis)
    max = np.max(x, axis=axis)
    return (x-min)/(max-min)

def z_normalize(x, axis=None):
    mean = np.mean(x, axis=axis)
    std = np.std(x, axis=axis)
    return (x - mean)/std

def save_data(data, result_dir):
    units = data['units']
    batch_size = data['batch_size']
    fname = 'units%d_bsize%d.pkl' % (units, batch_size)
    with open(os.path.join(result_dir, fname), 'wb') as f:
        pkl.dump(data, f)
