from keras.datasets import mnist, fashion_mnist
from keras.utils import to_categorical
from sklearn.datasets import load_digits
import numpy as np

class Mnist(object):
    def __init__(self):
        self.type = 'classification'
        self.num_classes = 10
        self.inputs = 784
        self.outputs = 10

    def load_data(self):
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train = x_train.astype(np.float32) / 255.
        x_test = x_test.astype(np.float32) / 255.
        x_train = x_train.reshape(-1,self.inputs)
        x_test = x_test.reshape(-1,self.inputs)
        return (x_train, y_train), (x_test, y_test)

class Fashion(object):
    def __init__(self):
        self.type = 'classification'
        self.num_classes = 10
        self.inputs = 784
        self.outputs = 10

    def load_data(self):
        (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
        x_train = x_train.astype(np.float32) / 255.
        x_test = x_test.astype(np.float32) / 255.
        x_train = x_train.reshape(-1,self.inputs)
        x_test = x_test.reshape(-1,self.inputs)
        return (x_train, y_train), (x_test, y_test)