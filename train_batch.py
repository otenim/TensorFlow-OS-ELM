import os
import argparse
import datasets
import models
from keras.optimizers import Adam

curdir = os.path.dirname(os.path.abspath(__file__))
parser = argparse.ArgumentParser()
parser.add_argument(
    'model',
    choices=[
        'mnist_mlp',
        'mnist_cnn',
        'fashion_mlp',
        'fashion_cnn'
    ])
parser.add_argument('--weights', default=None)
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=32)

def main(args):

    # instantiate model and dataset
    if args.model == 'mnist_mlp':
        model = models.create_mnist_mlp()
        dataset = datasets.get_dataset('mnist').load_data()
        lossfun = 'categorical_crossentropy'
    elif args.model == 'mnist_cnn':
        model = models.create_mnist_cnn()
        dataset = datasets.get_dataset('mnist').load_data(network='cnn')
        lossfun = 'categorical_crossentropy'
    elif args.model == 'fashion_mlp':
        model = models.create_fashion_mlp()
        dataset = datasets.get_dataset('mnist').load_data()
        lossfun = 'categorical_crossentropy'
    elif args.model == 'fashion_cnn':
        model = models.create_fashion_cnn()
        dataset = datasets.get_dataset('mnist').load_data(network='cnn')
        lossfun = 'categorical_crossentropy'
    else:
        raise Exception('unknown model was specified.')
    model.compile(optimizer=Adam(), loss=lossfun, metrics=['accuracy'])

    # prepare dataset
    (x_train, y_train), (x_test, y_test) = dataset

    # training
    model.fit(
        x=x_train,
        y=y_train,
        batch_size=args.batch_size,
        epochs=args.epochs,
        verbose=1,
        validation_data=(x_test,y_test),
        shuffle=True)

    # save weights
    if args.weights:
        if os.path.exists(args.weights) == False:
            os.makedirs(args.weights)
        fname = 'w_epochs%d_bsize%d.h5' % (args.epochs, args.batch_size)
        model.save_weights(os.path.join(args.weights, fname))

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
