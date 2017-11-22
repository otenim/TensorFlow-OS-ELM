import numpy as np
import os
import argparse
import models
import datasets
import time
from tqdm import tqdm
import utils

parser = argparse.ArgumentParser()
parser.add_argument(
    '--dataset',
    choices=['mnist', 'fashion', 'digits', 'boston'],
    default='mnist'
)
parser.add_argument('--units', type=int, default=1024)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--activation', choices=['sigmoid'], default='sigmoid')
parser.add_argument('--loss', choices=['mean_squared_error'], default='mean_squared_error')

def main(args):

    # prepare dataset
    dataset = datasets.get_dataset(args.dataset)
    (x_train, y_train), (x_test, y_test) = dataset.load_data()
    border = int(args.units * 1.1)
    x_train_init, x_train_seq = x_train[:border], x_train[border:]
    y_train_init, y_train_seq = y_train[:border], y_train[border:]

    # training
    os_elm = models.OS_ELM(
        inputs=dataset.inputs,
        units=args.units,
        outputs=dataset.outputs,
        activation=args.activation,
        loss=args.loss)

    print('now initial training phase...')
    os_elm.init_train(x_train_init, y_train_init)

    print('now sequential training phase...')
    pbar = tqdm(total=len(x_train_seq))
    for i in range(0, len(x_train_seq), args.batch_size):
        x_batch = x_train_seq[i:i+args.batch_size]
        y_batch = y_train_seq[i:i+args.batch_size]
        os_elm.seq_train(x_batch, y_batch)
        pbar.update(len(x_batch))
    pbar.close()

    print('now evaluation phase...')
    test_loss = os_elm.compute_loss(x_test,y_test)
    print('test loss: %f' % test_loss)
    if dataset.type == 'classification':
        test_acc = os_elm.compute_accuracy(x_test,y_test)
        print('test acc: %f' % test_acc)

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
