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
        y_train = to_categorical(y_train.astype(np.float32), self.num_classes)
        y_test = to_categorical(y_test.astype(np.float32), self.num_classes)
        return (x_train, y_train), (x_test, y_test)

class Mnist_inv(object):
    def __init__(self):
        self.type = 'classification'
        self.num_classes = 10
        self.inputs = 784
        self.outputs = 10

    def load_data(self):
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train = 1. - x_train.astype(np.float32) / 255.
        x_test = 1. - x_test.astype(np.float32) / 255.
        x_train = x_train.reshape(-1,self.inputs)
        x_test = x_test.reshape(-1,self.inputs)
        y_train = to_categorical(y_train.astype(np.float32), self.num_classes)
        y_test = to_categorical(y_test.astype(np.float32), self.num_classes)
        return (x_train, y_train), (x_test, y_test)

class Mnist_noise(object):
    def __init__(self, mean=0., sigma=0.4):
        self.type = 'classification'
        self.num_classes = 10
        self.inputs = 784
        self.outputs = 10
        self.mean = mean
        self.sigma = sigma

    def load_data(self):
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train = x_train.astype(np.float32) / 255.
        x_test = x_test.astype(np.float32) / 255.
        gauss = np.random.normal(self.mean, self.sigma, size=x_train.shape)
        x_train = np.clip(x_train+gauss, 0., 1.)
        gauss = np.random.normal(self.mean, self.sigma, size=x_test.shape)
        x_test = np.clip(x_test+gauss, 0., 1.)
        x_train = x_train.reshape(-1,self.inputs)
        x_test = x_test.reshape(-1,self.inputs)
        y_train = to_categorical(y_train.astype(np.float32), self.num_classes)
        y_test = to_categorical(y_test.astype(np.float32), self.num_classes)
        return (x_train, y_train), (x_test, y_test)

class Mnist_anomal(object):
    def __init__(self, mean=0., sigma=0.4):
        self.type = 'classification'
        self.num_classes = 10
        self.inputs = 784
        self.outputs = 10
        self.mean = mean
        self.sigma = sigma

    def load_data(self):
        mnist_inv = Mnist_inv()
        mnist_noise = Mnist_noise(self.mean,self.sigma)
        (x_train_inv, y_train_inv), (x_test_inv, y_test_inv) = mnist_inv.load_data()
        (x_train_noise, y_train_noise), (x_test_noise, y_test_noise) = mnist_noise.load_data()
        x_train = np.concatenate([x_train_inv,x_train_noise], axis=0)
        y_train = np.concatenate([y_train_inv,y_train_noise], axis=0)
        x_test = np.concatenate([x_test_inv,x_test_noise], axis=0)
        y_test = np.concatenate([y_test_inv,y_test_noise], axis=0)
        # shuffle
        perm = np.random.permutation(len(x_train))
        x_train = x_train[perm]
        y_train = y_train[perm]
        perm = np.random.permutation(len(x_test))
        x_test = x_test[perm]
        y_test = y_test[perm]
        return (x_train[:60000],y_train[:60000]),(x_test[:10000],y_test[:10000])

class Mnist_mini(object):
    def __init__(self):
        self.type = 'classification'
        self.num_classes = 10
        self.inputs = 64
        self.outputs = 10

    def load_data(self):
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train_resized = []
        x_test_resized = []
        for x in x_train:
            x = utils.resize_array(x, size=(8,8))
            x_train_resized.append(x)
        for x in x_test:
            x = utils.resize_array(x, size=(8,8))
            x_test_resized.append(x)
        x_train_resized = np.array(x_train_resized)
        x_test_resized = np.array(x_test_resized)
        del x_train
        del x_test
        x_train_resized = x_train_resized.astype(np.float32) / 255.
        x_test_resized = x_test_resized.astype(np.float32) / 255.
        x_train_resized = x_train_resized.reshape(-1,self.inputs)
        x_test_resized = x_test_resized.reshape(-1,self.inputs)
        y_train = to_categorical(y_train.astype(np.float32), self.num_classes)
        y_test = to_categorical(y_test.astype(np.float32), self.num_classes)
        return (x_train_resized, y_train), (x_test_resized, y_test)

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
        y_train = to_categorical(y_train.astype(np.float32), self.num_classes)
        y_test = to_categorical(y_test.astype(np.float32), self.num_classes)
        return (x_train, y_train), (x_test, y_test)

class Fashion_inv(object):
    def __init__(self):
        self.type = 'classification'
        self.num_classes = 10
        self.inputs = 784
        self.outputs = 10

    def load_data(self):
        (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
        x_train = 1. - x_train.astype(np.float32) / 255.
        x_test = 1. - x_test.astype(np.float32) / 255.
        x_train = x_train.reshape(-1,self.inputs)
        x_test = x_test.reshape(-1,self.inputs)
        y_train = to_categorical(y_train.astype(np.float32), self.num_classes)
        y_test = to_categorical(y_test.astype(np.float32), self.num_classes)
        return (x_train, y_train), (x_test, y_test)

class Fashion_noise(object):
    def __init__(self, mean=0., sigma=0.4):
        self.type = 'classification'
        self.num_classes = 10
        self.inputs = 784
        self.outputs = 10
        self.mean = mean
        self.sigma = sigma

    def load_data(self):
        (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
        x_train = x_train.astype(np.float32) / 255.
        x_test = x_test.astype(np.float32) / 255.
        gauss = np.random.normal(self.mean, self.sigma, size=x_train.shape)
        x_train = np.clip(x_train+gauss, 0., 1.)
        gauss = np.random.normal(self.mean, self.sigma, size=x_test.shape)
        x_test = np.clip(x_test+gauss, 0., 1.)
        x_train = x_train.reshape(-1,self.inputs)
        x_test = x_test.reshape(-1,self.inputs)
        y_train = to_categorical(y_train.astype(np.float32), self.num_classes)
        y_test = to_categorical(y_test.astype(np.float32), self.num_classes)
        return (x_train, y_train), (x_test, y_test)

class Fashion_anomal(object):
    def __init__(self, mean=0., sigma=0.4):
        self.type = 'classification'
        self.num_classes = 10
        self.inputs = 784
        self.outputs = 10
        self.mean = mean
        self.sigma = sigma

    def load_data(self):
        fashion_inv = Fashion_inv()
        fashion_noise = Fashion_noise(self.mean,self.sigma)
        (x_train_inv, y_train_inv), (x_test_inv, y_test_inv) = fashion_inv.load_data()
        (x_train_noise, y_train_noise), (x_test_noise, y_test_noise) = fashion_noise.load_data()
        x_train = np.concatenate([x_train_inv,x_train_noise], axis=0)
        y_train = np.concatenate([y_train_inv,y_train_noise], axis=0)
        x_test = np.concatenate([x_test_inv,x_test_noise], axis=0)
        y_test = np.concatenate([y_test_inv,y_test_noise], axis=0)
        # shuffle
        perm = np.random.permutation(len(x_train))
        x_train = x_train[perm]
        y_train = y_train[perm]
        perm = np.random.permutation(len(x_test))
        x_test = x_test[perm]
        y_test = y_test[perm]
        return (x_train[:60000],y_train[:60000]),(x_test[:10000],y_test[:10000])

class Fashion_mini(object):
    def __init__(self):
        self.type = 'classification'
        self.num_classes = 10
        self.inputs = 64
        self.outputs = 10

    def load_data(self):
        (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
        x_train_resized = []
        x_test_resized = []
        for x in x_train:
            x = utils.resize_array(x, size=(8,8))
            x_train_resized.append(x)
        for x in x_test:
            x = utils.resize_array(x, size=(8,8))
            x_test_resized.append(x)
        x_train_resized = np.array(x_train_resized)
        x_test_resized = np.array(x_test_resized)
        del x_train
        del x_test
        x_train_resized = x_train_resized.astype(np.float32) / 255.
        x_test_resized = x_test_resized.astype(np.float32) / 255.
        x_train_resized = x_train_resized.reshape(-1,self.inputs)
        x_test_resized = x_test_resized.reshape(-1,self.inputs)
        y_train = to_categorical(y_train.astype(np.float32), self.num_classes)
        y_test = to_categorical(y_test.astype(np.float32), self.num_classes)
        return (x_train_resized, y_train), (x_test_resized, y_test)

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
        x /= 16.
        y = digits.target.astype(np.float32)
        y = to_categorical(y, self.num_classes)
        border = int(len(x) * self.split)
        x_train, x_test = x[:border], x[border:]
        y_train, y_test = y[:border], y[border:]
        return (x_train, y_train), (x_test, y_test)

class Digits_inv(object):
    def __init__(self):
        self.type = 'classification'
        self.num_classes = 10
        self.inputs = 64
        self.outputs= 10
        self.split = 0.8

    def load_data(self):
        digits = load_digits()
        x = digits.data.astype(np.float32)
        x /= 16.
        x = 1.0 - x
        y = digits.target.astype(np.float32)
        y = to_categorical(y, self.num_classes)
        border = int(len(x) * self.split)
        x_train, x_test = x[:border], x[border:]
        y_train, y_test = y[:border], y[border:]
        return (x_train, y_train), (x_test, y_test)

class Digits_noise(object):
    def __init__(self, mean=0., sigma=0.4):
        self.type = 'classification'
        self.num_classes = 10
        self.inputs = 64
        self.outputs= 10
        self.split = 0.8
        self.mean = mean
        self.sigma = sigma

    def load_data(self):
        digits = load_digits()
        x = digits.data.astype(np.float32)
        x /= 16.
        gauss = np.random.normal(self.mean, self.sigma, size=x.shape)
        x = np.clip(x+gauss, 0., 1.)
        y = digits.target.astype(np.float32)
        y = to_categorical(y, self.num_classes)
        border = int(len(x) * self.split)
        x_train, x_test = x[:border], x[border:]
        y_train, y_test = y[:border], y[border:]
        return (x_train, y_train), (x_test, y_test)

class Digits_anomal(object):
    def __init__(self, mean=0., sigma=0.4):
        self.type = 'classification'
        self.num_classes = 10
        self.inputs = 64
        self.outputs= 10
        self.split = 0.8
        self.mean = mean
        self.sigma = sigma

    def load_data(self):
        digits_noise = Digits_noise(self.mean, self.sigma)
        digits_inv = Digits_inv()
        (x_train_noise,y_train_noise),(x_test_noise,y_test_noise) = digits_noise.load_data()
        (x_train_inv,y_train_inv),(x_test_inv,y_test_inv) = digits_inv.load_data()
        x_train = np.concatenate((x_train_noise,x_train_inv),axis=0)
        y_train = np.concatenate((y_train_noise,y_train_inv),axis=0)
        x_test = np.concatenate((x_test_noise,x_test_inv),axis=0)
        y_test = np.concatenate((y_test_noise,y_test_inv),axis=0)
        perm = np.random.permutation(len(x_train))
        x_train = x_train[perm]
        y_train = y_train[perm]
        perm = np.random.permutation(len(x_test))
        x_test = x_test[perm]
        y_test = y_test[perm]
        return (x_train[:1437], y_train[:1437]), (x_test[:360], y_test[:360])
