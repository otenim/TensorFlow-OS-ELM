import numpy as np
import argparse
import os
import tqdm
import datasets
import time
from sklearn.decomposition import PCA

parser = argparse.ArgumentParser()
parser.add_argument('--n_components', type=int, default=32)

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
    model = PCA(n_components=args.n_components)

    # ========================================================
    # Training
    # ========================================================
    s_time = time.time()
    model.fit(x_train_normal)
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
        h = model.transform(x)
        y = model.inverse_transform(h)
        # loss = np.mean((x - y)**2)
        loss = np.mean(np.abs(x - y))
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
