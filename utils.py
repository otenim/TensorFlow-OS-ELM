import numpy as np

# normalize dataset into [0, 1.0] along with given axis
def normalize_dataset(dataset, axis=0):
    max_value = np.max(dataset, axis=axis)
    min_value = np.min(dataset, axis=axis)
    normalized = (dataset - min_value) / (max_value - min_value)
    return normalized
