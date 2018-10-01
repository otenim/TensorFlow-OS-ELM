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

    num_ini_samples = int(args.units * 2)
    x_train_ini = x_train[:num_ini_samples]
    t_train_ini = t_train[:num_ini_samples]
    x_train_seq = x_train[num_ini_samples:]
    t_train_seq = t_train[num_ini_samples:]

    # instantiate model (oselm)
    model_oselm = OS_ELM(
        n_input_nodes=x_train.shape[-1],
        n_hidden_nodes=args.units,
        n_output_nodes=t_train.shape[-1],
    )

    # ini train (oselm)
    print(x_train_ini.shape)
    model_oselm.init_train(x_train_ini, t_train_ini)

    # seq train (oselm)
    steps_per_epoch = int(np.ceil(len(x_train_seq) / args.batch_size))
    stime = time.time()
    pbar = tqdm(total=steps_per_epoch)
    for i in range(0, len(x_train), args.batch_size):
        x = x_train_seq[i : i + args.batch_size]
        t = t_train_seq[i : i + args.batch_size]
        model_oselm.seq_train(x, t)
        pbar.update()
    pbar.close()
    etime = time.time()
    print('training time: %.5f [sec]' % (etime - stime))

    # eval (oselm)
    steps_per_epoch = int(np.ceil(len(x_test)) / args.batch_size)
    pbar = tqdm(total=steps_per_epoch)
    acc = 0.
    for i in range(0, len(x_test), args.batch_size):
        x = x_test[i : i + args.batch_size]
        t = t_test[i : i + args.batch_size]
        acc += model_oselm.evaluate(x, t, metrics=['accuracy'])[0]
        pbar.update()
    pbar.close()
    acc /= steps_per_epoch
    print('validation accuracy: %.5f' % acc)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
