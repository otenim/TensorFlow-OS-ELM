from keras.datasets import mnist, fashion_mnist, boston_housing
from keras.utils import to_categorical
from sklearn.datasets import load_digits
from utils import min_max_normalize, z_normalize
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
        y_train = to_categorical(y_train.astype(np.float32), self.num_classes)
        y_test = to_categorical(y_test.astype(np.float32), self.num_classes)
        return (x_train, y_train), (x_test, y_test)

class Fashion(object):
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
        y_train = to_categorical(y_train.astype(np.float32), self.num_classes)
        y_test = to_categorical(y_test.astype(np.float32), self.num_classes)
        return (x_train, y_train), (x_test, y_test)

class Digits(object):
    def __init__(self):
        self.type = 'classification'
        self.num_classes = 10
        self.inputs = 64
        self.outputs= 10
        self.split = 0.8

    def load_data(self):
        digits = load_digits()
        x = digits.data.astype(np.float32)
        x /= 255.
        y = digits.target.astype(np.float32)
        y = to_categorical(y, self.num_classes)
        border = int(len(x) * self.split)
        x_train, x_test = x[:border], x[border:]
        y_train, y_test = y[:border], y[border:]
        return (x_train, y_train), (x_test, y_test)

class Boston(object):
    def __init__(self):
        self.type = 'regression'
        self.inputs = 13
        self.outputs = 1

    def load_data(self):
        (x_train, y_train), (x_test, y_test) = boston_housing.load_data()
        x_train = z_normalize(x_train.astype(np.float32), axis=0)
        x_test = z_normalize(x_test.astype(np.float32), axis=0)
        y_train = y_train.astype(np.float32)
        y_test = y_test.astype(np.float32)
        return (x_train,y_train), (x_test, y_test)

def get_dataset(dataset_name):
    if dataset_name == 'mnist':
        dataset = Mnist()
    elif dataset_name == 'fashion':
        dataset = Fashion()
    elif dataset_name == 'digits':
        dataset = Digits()
    elif dataset_name == 'boston':
        dataset = Boston()
    else:
        raise Exception('unknown dataset was specified.')
    return dataset
