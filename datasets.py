from keras.datasets import mnist
from keras.utils import to_categorical
import numpy as np

def load_mnist():
    num_classes = 10
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.astype(np.float32) / 255.
    x_test = x_test.astype(np.float32) / 255.
    y_train = to_categorical(y_train.astype(np.float32), num_classes)
    y_test = to_categorical(y_test.astype(np.float32), num_classes)
    return (x_train, y_train), (x_test, y_test)
