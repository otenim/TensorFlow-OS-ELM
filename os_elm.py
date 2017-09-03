import numpy as np

# Network definition
class OS_ELM(object):

    def _mean_squared_error(self, out, y):
        batch_size = len(out)
        return 0.5 * np.sum((out - y)**2) / batch_size

    def _accuracy(self, out, y):
        batch_size = len(out)
        accuracy = np.sum((np.argmax(out, axis=1) == np.argmax(y, axis=1)))
        return 100.0 * accuracy / batch_size

    def _sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-1.0 * x))

    def _relu(self, x):
        return np.maximum(0, x)

    def _softmax(self, x):
        c = np.max(x, axis=1).reshape(-1, 1)
        upper = np.exp(x - c)
        lower = np.sum(upper, axis=1).reshape(-1, 1)
        return upper / lower

    def __init__(self, inputs, units, outputs):
        self.inputs = inputs
        self.units = units
        self.outputs = outputs
        self.alpha = np.random.rand(inputs, units) * 2.0 - 1.0 # [-1.0, 1.0]
        self.beta = np.random.rand(units, outputs) * 2.0 - 1.0
        self.bias = np.random.rand(1, units) * 2.0 - 1.0
        self.P = None
        self.is_init_phase = True
        self.is_seq_phase = False

    def __call__(self, x):
        h1 = x.dot(self.alpha) + self.bias
        a1 = self._sigmoid(h1)
        h2 = a1.dot(self.beta)
        return h2

    def eval(self, x, y, mode='classify'):
        out = self(x)
        loss = self._mean_squared_error(out, y)
        if mode == 'classify':
            accuracy = self._accuracy(out, y)
            return loss, accuracy
        elif mode == 'regress':
            return loss
        else:
            raise Exception('unnexpected mode [%s] was entered.' % mode)


    def init_train(self, x0, y0):
        assert self.is_init_phase, 'initial training phase was over. use [seq_train] instead of [init_train]'
        assert len(x0) >= self.units, 'initial dataset length must be >= %d' % (self.units)
        H0 = self._sigmoid(x0.dot(self.alpha) + self.bias)
        H0T = H0.T
        self.P = np.linalg.pinv(H0T.dot(H0))
        self.beta = self.P.dot(H0T).dot(y0)
        self.is_init_phase = False
        self.is_seq_phase = True

    def seq_train(self, x, y):
        assert self.is_seq_phase, 'you have not finished the initial training phase yet'
        H = self._sigmoid(x.dot(self.alpha))
        HT = H.T
        I = np.eye(len(x))    # I.shape = (N, N) N:length of inputa data

        # update P
        temp = np.linalg.pinv(I + H.dot(self.P).dot(HT))    # temp.shape = (N, N)
        self.P = self.P - (self.P.dot(HT).dot(temp).dot(H).dot(self.P))

        # update beta
        self.beta = self.beta + (self.P.dot(HT).dot(y - H.dot(self.beta)))
