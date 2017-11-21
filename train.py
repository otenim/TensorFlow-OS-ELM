import numpy as np
import os
import argparse
import models
import datasets
import time
from tqdm import tqdm
from utils import save_data

curdir = os.path.dirname(os.path.abspath(__file__))
parser = argparse.ArgumentParser()
parser.add_argument('--result', default=None)
parser.add_argument('--weights', default=None)
parser.add_argument(
    '--dataset',
    choices=['mnist', 'fashion', 'digits', 'boston'],
    default='mnist'
)
parser.add_argument('--epochs', type=int, default=10)
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

    seq_train_time_data = []
    init_train_time_data = []
    pred_time_data = []
    test_loss_data = []
    test_acc_data = []
    # training loop
    for epoch in range(args.epochs):

        os_elm = models.OS_ELM(
            inputs=dataset.inputs,
            units=args.units,
            outputs=dataset.outputs,
            activation=args.activation,
            loss=args.loss)

        print('********** Epoch (%d/%d) **********' % (epoch+1,args.epochs))
        print('initial training phase...')
        start = time.time()
        os_elm.init_train(x_train_init, y_train_init)
        init_train_time_data.append(time.time() - start)

        print('sequential training phase...')
        times = []
        pbar = tqdm(total=len(x_train_seq))
        for i in range(0, len(x_train_seq), args.batch_size):
            x_batch = x_train_seq[i:i+args.batch_size]
            y_batch = y_train_seq[i:i+args.batch_size]
            start = time.time()
            os_elm.seq_train(x_batch, y_batch)
            seq_train_time_data.append(time.time() - start)
            pbar.update(len(x_batch))
        pbar.close()

        print('evaluation phase...')
        start = time.time()
        out = os_elm(x_test[:args.batch_size])
        pred_time_data.append(time.time() - start)

        test_loss = os_elm.compute_loss(x_test,y_test)
        test_loss_data.append(test_loss)
        print('test loss: %f' % test_loss)
        if dataset.type == 'classification':
            test_acc = os_elm.compute_accuracy(x_test,y_test)
            test_acc_data.append(test_acc)
            print('test acc: %f' % test_acc)

    # save results
    if args.result:
        data = {
            'dataset': args.dataset,
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'units': args.units,
            'activation': args.activation,
            'loss': args.loss,
            'mean_init_train_time': np.mean(init_train_time_data),
            'mean_seq_train_time': np.mean(seq_train_time_data),
            'mean_test_loss': np.mean(test_loss_data),
            'mean_pred_time': np.mean(pred_time_data)}
        if dataset.type == 'classification':
            data['mean_test_acc'] = np.mean(test_acc_data)
        if os.path.exists(args.result) == False:
            os.makedirs(args.result)
        save_data(data, args.result)

    # save weights
    if args.weights:
        if os.path.exists(args.weights) == False:
            os.makedirs(args.weights)
        fname = 'w_%s_units%d_bsize%d.pkl' % (args.dataset, args.units, args.batch_size)
        os_elm.save_weights(os.path.join(args.weights, fname))


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
