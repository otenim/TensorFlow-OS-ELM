import keras
from keras.models import Model
from keras.layers import Input
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.optimizers import Adam
from keras.losses import categorical_crossentropy
import numpy as np
import pickle

# Network definition
class OS_ELM(object):

    def __mean_squared_error(self, out, y):
        return 0.5 * np.mean((out - y)**2)

    def __accuracy(self, out, y):
        batch_size = len(out)
        accuracy = np.sum((np.argmax(out, axis=1) == np.argmax(y, axis=1)))
        return accuracy / batch_size

    def __sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-1.0 * x))

    def __relu(self, x):
        return np.maximum(0, x)

    def __softmax(self, x):
        c = np.max(x, axis=1).reshape(-1, 1)
        upper = np.exp(x - c)
        lower = np.sum(upper, axis=1).reshape(-1, 1)
        return upper / lower

    def __init__(self, inputs, units, outputs, activation='sigmoid', loss='mean_squared_error'):
        self.inputs = inputs
        self.units = units
        self.outputs = outputs
        self.alpha = np.random.rand(inputs, units) * 2.0 - 1.0 # [-1.0, 1.0]
        self.beta = np.random.rand(units, outputs) * 2.0 - 1.0 # [-1.0, 1.0]
        self.bias = np.zeros(shape=(1,self.units))
        self.p = None
        if loss == 'mean_squared_error':
            self.lossfun = self.__mean_squared_error
        else:
            raise Exception('unknown loss function was specified.')
        if activation == 'sigmoid':
            self.actfun = self.__sigmoid
        elif activation == 'relu':
            self.actfun = self.__relu
        else:
            raise Exception('unknown activation function was specified.')

    def __call__(self, x):
        h1 = x.dot(self.alpha) + self.bias
        a1 = self.actfun(h1)
        h2 = a1.dot(self.beta)
        out = self.__softmax(h2)
        return out

    def compute_accuracy(self, x, y):
        return self.__accuracy(self(x), y)

    def compute_loss(self, x, y):
        return self.lossfun(self(x), y)

    def init_train(self, x, y):
        assert len(x) >= self.units, 'initial dataset size must be >= %d' % (self.units)
        H = self.actfun(x.dot(self.alpha) + self.bias)
        HT = H.T
        self.p = np.linalg.pinv(HT.dot(H))
        self.beta = self.p.dot(HT).dot(y)

    def seq_train(self, x, y):
        H = self.actfun(x.dot(self.alpha))
        HT = H.T
        I = np.eye(len(x))# I.shape = (N, N) N:length of inputa data

        # update P
        temp = np.linalg.pinv(I + H.dot(self.p).dot(HT))    # temp.shape = (N, N)
        self.p = self.p - (self.p.dot(HT).dot(temp).dot(H).dot(self.p))

        # update beta
        self.beta = self.beta + (self.p.dot(HT).dot(y - H.dot(self.beta)))

    def save_weights(self, path):
        weights = {
            'alpha': self.alpha,
            'beta': self.beta,
            'p': self.p}
        with open(path, 'wb') as f:
            pickle.dump(weights, f)

    def load_weights(self, path):
        with open(path, 'rb') as f:
            weights = pickle.load(f)
            self.alpha = weights['alpha']
            self.beta = weights['beta']
            self.p = weights['p']

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self,f)

def create_mnist_mlp():
    input_shape = (28**2,)
    num_classes = 10
    input = Input(shape=input_shape)
    x = Dense(512, activation='relu')(input)
    x = Dropout(0.2)(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(num_classes, activation='softmax')(x)
    model = Model(input, x)
    return model

def create_mnist_cnn():
    input_shape = (28,28,1)
    num_classes = 10
    input = Input(shape=input_shape)
    x = Conv2D(32, (3,3), activation='relu')(input)
    x = Conv2D(64, (3,3), activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.25)(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(num_classes, activation='softmax')(x)
    model = Model(input,x)
    return model

def create_fashion_mlp():
    return create_mnist_mlp()

def create_fashion_cnn():
    return create_mnist_cnn()

def get_batch_model(model_name):
    if model_name == 'mnist_mlp':
        model = create_mnist_mlp()
        opt = Adam()
        lossfun = categorical_crossentropy
    elif model_name == 'mnist_cnn':
        model = create_mnist_cnn()
        opt = Adam()
        lossfun = categorical_crossentropy
    elif model_name == 'fashion_mlp':
        model = create_fashion_mlp()
        opt = Adam()
        lossfun = categorical_crossentropy
    elif model_name == 'fashion_cnn':
        model = create_fashion_cnn()
        opt = Adam()
        lossfun = categorical_crossentropy
    else:
        raise Exception('unknown model was specified.')
    return (model, opt, lossfun)
