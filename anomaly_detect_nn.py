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
parser.add_argument('--abnormal_ratio', type=float, default=0.1)
parser.add_argument('--loss',choices=[
    'mean_squared_error',
    'mean_absolute_error',
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
    x_train_normal = x_train_normal[np.random.permutation(len(x_train_normal))]
    x_test_normal = x_test_normal[np.random.permutation(len(x_test_normal))]
    x_test_abnormal = x_test_abnormal[np.random.permutation(len(x_test_abnormal))]
    x_valid = x_test_normal[:len(x_test_normal) // 2]
    x_eval_normal = x_test_normal[len(x_test_normal) // 2:]
    x_eval_abnormal = x_test_abnormal[:int(len(x_eval_normal) * args.abnormal_ratio)]

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
    # compute mean and sigma
    pbar = tqdm.tqdm(total=len(x_valid), desc='computing mean and sigma')
    losses = []
    for x in x_valid:
        x = np.expand_dims(x, axis=0)
        [loss] = model.test_on_batch(x, x, metrics=['loss'])
        losses.append(loss)
        pbar.update(n=1)
    pbar.close()
    mean = np.mean(losses)
    sigma = np.std(losses)
    print('mean: %f' % mean)
    print('sigma: %f' % sigma)

    # evaluation
    pbar = tqdm.tqdm(total=len(x_eval_normal)+len(x_eval_abnormal), desc='evaluating anomaly detection accuracy')
    x_eval = np.concatenate((x_eval_normal, x_eval_abnormal), axis=0)
    losses = []
    labels = np.concatenate(([False] * len(x_eval_normal), [True] * len(x_eval_abnormal)))
    for x in x_eval:
        x = np.expand_dims(x, axis=0)
        [loss] = model.test_on_batch(x, x, metrics=['loss'])
        losses.append(loss)
        pbar.update(n=1)
    pbar.close()
    losses = np.array(losses)
    losses = (losses - mean) / sigma

    # compute precision, recall, f-measure for each k
    ks = [1., 2., 3., 4., 5., 6., 7., 8., 9., 10.]
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
