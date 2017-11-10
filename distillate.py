import keras
from keras.models import Model
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.datasets import mnist
import numpy as np
import models
import argparse
import os
from tqdm import tqdm

curdir = os.path.dirname(os.path.abspath(__file__))
parser = argparse.ArgumentParser()
parser.add_argument('weight')
parser.add_argument('--epochs', type=int, default=20)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--units', type=int, default=1024)

def main(args):

    deep_model = models.create_mnist_model()
    deep_model.load_weights(args.weight)
    
    os_elm = models.OS_ELM(
        inputs=28**2,
        units=args.units,
        outputs=10,
        activation='sigmoid')

    (x_train, _), (x_test, y_test) = mnist.load_data()
    x_train = x_train.astype(np.float32) / 255.
    x_train = x_train.reshape(-1,28**2)
    x_test = x_test.astype(np.float32) / 255.
    x_test = x_test.reshape(-1,28**2)
    y_test = to_categorical(y_test).astype(np.float32)

    init_batch_size = int(args.units * 1.2)
    x_train_init = x_train[:init_batch_size,:].reshape(-1,28**2)
    x_train_seq = x_train[init_batch_size:,:].reshape(-1,28**2)

    # Initial training
    print('Initial training on %d images' % len(x_train_init))
    y_train_init = deep_model.predict(x_train_init)
    os_elm.init_train(x_train_init, y_train_init)
    loss, acc = os_elm.eval(x_test, y_test)
    print(loss)
    print(acc)


    # Sequential training
    print('Sequential training on %d images' % len(x_train_seq))
    pbar = tqdm(total=len(x_train_seq))
    for i in range(0, len(x_train_seq), args.batch_size):
        x = x_train_seq[i:i+args.batch_size]
        y = deep_model.predict(x)
        os_elm.seq_train(x, y)
        pbar.update(len(x))
    pbar.close()

    loss, acc = os_elm.eval(x_test, y_test)
    print('final accuracy: %f' % (acc))
    print('final loss: %f' % (loss))

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
