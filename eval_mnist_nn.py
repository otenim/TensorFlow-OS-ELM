import os
import argparse
import time
import numpy as np
from keras.datasets import mnist
from keras.utils import to_categorical
from nn import NN
from os_elm import OS_ELM
from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument('--units', type=int, default=1024)
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=32)


def main(args):

    # load dataset
    (x_train, t_train), (x_test, t_test) = mnist.load_data()
    x_train = x_train.astype(dtype=np.float32) / 255.
    x_train = x_train.reshape(-1, 28**2)
    x_test = x_test.astype(dtype=np.float32) / 255.
    x_test = x_test.reshape(-1, 28**2)
    t_train = to_categorical(t_train, num_classes=10)
    t_test = to_categorical(t_test, num_classes=10)
    perm = np.random.permutation(len(x_train))
    x_train = x_train[perm]
    t_train = t_train[perm]
    perm = np.random.permutation(len(x_test))
    x_test = x_test[perm]
    t_test = t_test[perm]

    # instantiate model (nn)
    model_nn = NN(
        n_input_nodes=x_train.shape[-1],
        n_hidden_nodes=args.units,
        n_output_nodes=t_train.shape[-1],
        output_activation='softmax',
    )

    # train (nn)
    steps_per_epoch = int(np.ceil(len(x_train) / args.batch_size))
    stime = time.time()
    for epoch in range(args.epochs):
        print('%d/%d' % (epoch, args.epochs))
        pbar = tqdm(total=steps_per_epoch)
        for i in range(0, len(x_train), args.batch_size):
            x = x_train[i : i + args.batch_size]
            t = t_train[i : i + args.batch_size]
            model_nn.train_on_batch(x, t)
            pbar.update()
        pbar.close()
    etime = time.time()
    print('training time: %.5f [sec]' % (etime - stime))

    # eval (nn)
    steps_per_epoch = int(np.ceil(len(x_test)) / args.batch_size)
    pbar = tqdm(total=steps_per_epoch)
    acc = 0.
    for i in range(0, len(x_test), args.batch_size):
        x = x_test[i : i + args.batch_size]
        t = t_test[i : i + args.batch_size]
        acc += model_nn.test_on_batch(x, t, metrics=['accuracy'])[0]
        pbar.update()
    pbar.close()
    acc /= steps_per_epoch
    print('validation accuracy: %.5f' % acc)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
