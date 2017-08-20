import argparse
import chainer
import numpy as np
from matplotlib import pyplot
import os
import pickle
import time

current_directory = os.path.dirname(os.path.abspath(__file__))
parser = argparse.ArgumentParser()
parser.add_argument(
    '--batch_size',
    type=int,
    default=32,
)
parser.add_argument(
    '--epochs',
    type=int,
    default=20,
)
parser.add_argument(
    '--units',
    type=int,
    default=1000,
)

# Network definition
class OS_ELM(object):

    def _categorical_cross_entropy_error(self, out, y):
        batch_size = len(out)
        return -np.sum(y * np.log(out + 1.0e-7)) / batch_size

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

    def eval(self, x, y):
        out = self(x)
        loss = self._mean_squared_error(out, y)
        accuracy = self._accuracy(out, y)
        return  loss, accuracy

    def predict(self, x):
        out = self(x)
        return np.argmax(out, axis=1)

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

def main(args):

    # hyper parameters
    units = args.units
    batch_size = args.batch_size
    init_batch_size = int(units * 1.2)
    epochs = args.epochs
    inputs = 784
    outputs = 10

    # prepare datasets
    print("=====> preparing dataset")
    train, test = chainer.datasets.get_mnist()
    x_train, y_train = train._datasets[0], train._datasets[1]
    x_test, y_test = test._datasets[0], test._datasets[1]

    # preprocessing for input data (normalize [-1, 1])
    # x_train = x_train * 2.0 - 1.0
    # x_test = x_test * 2.0 - 1.0

    # translate labels into one-hot vectors
    y_test = np.eye(outputs)[y_test]
    y_train = np.eye(outputs)[y_train]

    # separate training data into two groups
    # (for initial training and for sequential training)
    x_train_init = x_train[:init_batch_size]
    y_train_init = y_train[:init_batch_size]
    x_train_seq = x_train[init_batch_size:]
    y_train_seq = y_train[init_batch_size:]


    # instantiate model
    print("=====> creating model")
    model = OS_ELM(inputs=inputs, units=units, outputs=outputs)

    # initial training phase
    print("=====> initial training phase")
    model.init_train(x_train_init, y_train_init)

    # sequential training phase
    print("=====> sequential training phase")
    n_train_seq = len(x_train_seq)
    n_test = len(x_test)
    plot_loss = []
    plot_val_loss = []
    plot_acc = []
    plot_val_acc = []
    plot_durations = []

    for epoch in range(epochs):
        print("epoch %d ----------" % (epoch + 1))

        # shuffle data
        idx = np.random.permutation(n_train_seq)
        x_train_seq = x_train_seq[idx]
        y_train_seq = y_train_seq[idx]

        start = time.time()
        for i in range(0, n_train_seq, batch_size):
            x, y = x_train_seq[i:i+batch_size], y_train_seq[i:i+batch_size]
            model.seq_train(x, y)
        duration = time.time() - start
        print('Elapsed time: %f[sec]' % (duration))

        # evaluation
        loss, acc = model.eval(x_train_seq[:n_test], y_train_seq[:n_test])
        print("main loss: %f" % (loss))
        print("main accuracy: %f%%" % (acc))

        val_loss, val_acc = model.eval(x_test, y_test)
        print("validation loss: %f" % (val_loss))
        print("validation accuracy: %f%%" % (val_acc))

        # add plot data
        plot_loss.append(loss)
        plot_acc.append(acc)
        plot_val_loss.append(val_loss)
        plot_val_acc.append(val_acc)
        plot_durations.append(duration)

    # save plot data as pickle file
    print("=====> saveing plot data...")
    plot = {
        'loss': plot_loss,
        'acc': plot_acc,
        'val_loss': plot_val_loss,
        'val_acc': plot_val_acc,
        'duration': plot_durations,
    }
    filename = 'oselm_batch' + str(batch_size) + '_epochs' + str(epochs) + '.dump'
    with open(os.path.join(current_directory, filename), 'wb') as f:
        pickle.dump(plot, f)

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
