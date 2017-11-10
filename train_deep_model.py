import keras
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.datasets import mnist
import numpy as np
import models
import argparse
import os

curdir = os.path.dirname(os.path.abspath(__file__))
parser = argparse.ArgumentParser()
parser.add_argument('--result', default=curdir)
parser.add_argument('--epochs', type=int, default=20)
parser.add_argument('--batch_size', type=int, default=32)

def main(args):

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.astype(np.float32) / 255.
    x_test = x_test.astype(np.float32) / 255.
    x_train = x_train.reshape(-1,28**2)
    x_test = x_test.reshape(-1,28**2)
    y_train = to_categorical(y_train).astype(np.float32)
    y_test = to_categorical(y_test).astype(np.float32)

    deep_model = models.create_mnist_model()
    deep_model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

    deep_model.fit(
        x=x_train,
        y=y_train,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_data=(x_test, y_test))

    if os.path.exists(args.result) == False:
        os.makedirs(args.result)
    deep_model.save_weights(os.path.join(args.result, 'deep_weight.h5'))



if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
