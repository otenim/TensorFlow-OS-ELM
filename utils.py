import numpy as np
import os

# normalize dataset into [0, 1.0] along with given axis
def normalize_dataset(dataset, axis=None):
    max_value = np.max(dataset, axis=axis)
    min_value = np.min(dataset, axis=axis)
    normalized = (dataset - min_value) / (max_value - min_value)
    return normalized

def dump_dataset(path, dataset):
    # dataset assumes 2D array
    dim = len(dataset.shape)
    with open(path, 'w') as f:
        if dim == 2:
            for i in range(len(dataset)):
                for j in range(len(dataset[0])):
                    f.write('%f ' % dataset[i][j])
                f.write('\n')
        elif dim == 1:
            for i in range(len(dataset)):
                f.write('%f\n' % dataset[i])
        else:
            Exception('Dimension of the input array must be 1 or 2')
