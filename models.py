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
        out = a1.dot(self.beta)
        return out

    def forward(self, x):
        return self(x)

    def classify(self, x):
        return self.__softmax(self(x))

    def compute_accuracy(self, x, y):
        out = self(x)
        acc = self.__accuracy(out, y)
        return acc

    def compute_loss(self, x, y):
        out = self(x)
        loss = self.lossfun(out,y)
        return loss

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
