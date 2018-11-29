import numpy as np
import argparse
import os
import tqdm
import datasets
import time
from nn import NN

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--batch_size',type=int,default=64)
parser.add_argument('--n_hidden_nodes', type=int, default=32)
parser.add_argument('--loss',choices=[
    'mean_squared_error',
    'mean_absolute_error',
    'softmax_cross_entropy'
],default='mean_absolute_error')
parser.add_argument('--hidden_activation', choices=[
    'relu', 'sigmoid', 'linear'], default='relu')
parser.add_argument('--output_activation', choices=[
    'relu', 'sigmoid', 'linear'], default='sigmoid')

def main(args):

    # ========================================================
    # Prepare datasets
    # ========================================================
    dataset_abnormal = datasets.Fashion()
    dataset_normal = datasets.Mnist()
    (x_train_normal, _), (x_test_normal, _) = dataset_normal.load_data()
    (_, _), (x_test_abnormal, _) = dataset_abnormal.load_data()

    # ========================================================
    # Build model
    # ========================================================
    model = NN(
        n_input_nodes=dataset_normal.inputs,
        n_hidden_nodes=args.n_hidden_nodes,
        n_output_nodes=dataset_normal.inputs,
        loss=args.loss,
        hidden_activation=args.hidden_activation,
        output_activation=args.output_activation,
    )

    # ========================================================
    # Training
    # ========================================================
    s_time = time.time()
    for epoch in range(args.epochs):
        pbar = tqdm.tqdm(total=len(x_train_normal), desc='Epoch (%d/%d)' % (epoch+1, args.epochs))
        for i in range(0, len(x_train_normal), args.batch_size):
            x_batch = x_train_normal[i:i+args.batch_size]
            model.train_on_batch(x_batch, x_batch)
            pbar.update(n=len(x_batch))
        pbar.close()
    e_time = time.time()
    print('training time: %f [sec]' % (e_time - s_time))

    # ========================================================
    # Evaluation (precision, recall, f-measure)
    # ========================================================
    pbar = tqdm.tqdm(
        total=(len(x_test_normal)+len(x_test_abnormal)),
        desc='now evaluating...',
    )

    # compute losses
    x_test = np.concatenate((x_test_normal, x_test_abnormal), axis=0)
    losses = []
    labels = np.concatenate(([False] * len(x_test_normal), [True] * len(x_test_abnormal)))
    for x in x_test:
        x = np.expand_dims(x, axis=0)
        [loss] = model.test_on_batch(x, x, metrics=['loss'])
        losses.append(loss)
        pbar.update(n=1)
    losses = np.array(losses)
    pbar.close()

    # normalize the loss values
    losses_normal = losses[:len(x_test_normal)]
    losses_abnormal = losses[len(x_test_normal):]
    mean = np.mean(losses_normal)
    sigma = np.std(losses_normal)
    losses = (losses - mean) / sigma

    # compute precision, recall, f-measure for each k (1., 2., 3.)
    ks = [1., 2., 3.]
    for k in ks:
        TP = np.sum(labels & (losses > k))
        precision = TP / np.sum(losses > k)
        recall = TP / np.sum(labels)
        f_measure = (2. * recall * precision) / (recall + precision)
        print('========== k = %.1f ==========' % k)
        print('precision: %.4f' % precision)
        print('recall: %.4f' % recall)
        print('f-measure: %.4f' % f_measure)

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
