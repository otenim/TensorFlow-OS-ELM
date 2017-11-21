import os
import argparse
import datasets
import models
from keras.optimizers import Adam

curdir = os.path.dirname(os.path.abspath(__file__))
parser = argparse.ArgumentParser()
parser.add_argument(
    'model',
    choices=['mnist_mlp','mnist_cnn','fashion_mlp','fashion_cnn'])
parser.add_argument(
    'dataset',
    choices=['mnist','fashion'])
parser.add_argument('--epochs', type=int, default=20)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--save_model', default=None)


def main(args):

    # instantiate model and dataset
    model, opt, lossfunc = models.get_batch_model(args.model)
    model.compile(optimizer=opt, loss=lossfunc)

    # prepare dataset
    dataset = datasets.get_dataset(args.dataset)
    (x_train, y_train), (x_test, y_test) = dataset.load_data()
    if args.model == 'mnist_cnn' or args.model == 'fashion_cnn':
        x_train = x_train.reshape(-1,28,28,1)
        x_test = x_test.reshape(-1,28,28,1)

    # training
    model.fit(
        x=x_train,
        y=y_train,
        batch_size=args.batch_size,
        epochs=args.epochs,
        verbose=1,
        validation_data=(x_test,y_test),
        shuffle=True)

    # save model
    if args.save_model:
        if os.path.exists(args.save_model) == False:
            os.makedirs(args.save_model)
        fname = 'm_%s_e%d_b%d.h5' % (args.dataset, args.epochs, args.batch_size)
        model.save(os.path.join(args.save_model, fname))

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
