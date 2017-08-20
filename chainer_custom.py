#!/usr/bin/env python
"""Fully-connected neural network example using MNIST dataset
This code is a custom loop version of train_mnist.py. That is, we train
models without using the Trainer class in chainer and instead write a
training loop that manually computes the loss of minibatches and
applies an optimizer to update the model.
"""
from __future__ import print_function
import argparse
import chainer
from chainer.dataset import convert
import chainer.links as L
import chainer.functions as F
from chainer import serializers
import time
import pickle
import ime
import os

# Network definition
class MLP(chainer.Chain):

    def __init__(self, n_units, n_out):
        super(MLP, self).__init__()
        with self.init_scope():
            # the size of the inputs to each layer will be inferred
            self.l1 = L.Linear(None, n_units)  # n_in -> n_units
            self.l2 = L.Linear(None, n_out)  # n_units -> n_out

    def __call__(self, x):
        h1 = F.relu(self.l1(x))
        return self.l2(h1)

current_directory = os.path.dirname(os.path.abspath(__file__))

def main():
    parser = argparse.ArgumentParser(description='Chainer example: MNIST')
    parser.add_argument('--batchsize', '-b', type=int, default=100,
                        help='Number of images in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=20,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--resume', '-r', default='',
                        help='Resume the training from snapshot')
    parser.add_argument('--unit', '-u', type=int, default=1000,
                        help='Number of units')
    args = parser.parse_args()

    print('GPU: {}'.format(args.gpu))
    print('# unit: {}'.format(args.unit))
    print('# Minibatch-size: {}'.format(args.batchsize))
    print('# epoch: {}'.format(args.epoch))
    print('')

    # Set up a neural network to train
    model = L.Classifier(MLP(args.unit, 10))
    if args.gpu >= 0:
        # Make a speciied GPU current
        chainer.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()  # Copy the model to the GPU

    # Setup an optimizer
    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)

    # Load the MNIST dataset
    train, test = chainer.datasets.get_mnist()

    train_count = len(train)
    test_count = len(test)

    train_iter = chainer.iterators.SerialIterator(train, args.batchsize)
    test_iter = chainer.iterators.SerialIterator(test, args.batchsize,
                                                 repeat=False, shuffle=False)

    # plot data
    sum_accuracy = 0
    sum_loss = 0
    plot_loss = []
    plot_val_loss = []
    plot_acc = []
    plot_val_acc = []
    plot_durations = []

    while train_iter.epoch < args.epoch:
        batch = train_iter.next()
        x_array, t_array = convert.concat_examples(batch, args.gpu)
        x = chainer.Variable(x_array)
        t = chainer.Variable(t_array)
        optimizer.update(model, x, t)
        sum_loss += float(model.loss.data) * len(t.data)
        sum_accuracy += float(model.accuracy.data) * len(t.data)

        if train_iter.is_new_epoch:
            print('epoch: {}', train_iter.epoch)

            loss = sum_loss / train_count
            acc = sum_accuracy / train_count
            plot_loss.append(loss)
            plot_acc.append(acc)
            print('train mean loss: {}, accuracy: {}'.format(loss, acc))


            # evaluation
            sum_accuracy = 0
            sum_loss = 0
            for batch in test_iter:
                x_array, t_array = convert.concat_examples(batch, args.gpu)
                x = chainer.Variable(x_array)
                t = chainer.Variable(t_array)
                loss = model(x, t)
                sum_loss += float(loss.data) * len(t.data)
                sum_accuracy += float(model.accuracy.data) * len(t.data)

            test_iter.reset()
            val_loss = sum_loss / test_count
            val_acc = sum_accuracy / test_count
            plot_val_loss.append(val_loss)
            plot_val_acc.append(val_acc)
            print('test mean  loss: {}, accuracy: {}'.format(val_loss, val_acc))
            sum_accuracy = 0
            sum_loss = 0

    # save plot data as pickle file
    plot = {
        'loss': plot_loss,
        'acc': plot_acc,
        'val_loss': plot_val_loss,
        'val_acc': plot_val_acc,
    }
    filename = 'chainer_batch' + str(args.batchsize) + '_epochs' + str(args.epoch) + '.dump'
    with open(os.path.join(current_directory, filename), 'wb') as f:
        pickle.dump(plot, f)


if __name__ == '__main__':
    main()
