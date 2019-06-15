import numpy as np
import argparse
import os
import tqdm
import datasets
import time
import random
from os_elm import OS_ELM
from sklearn.metrics import roc_auc_score

parser = argparse.ArgumentParser()
parser.add_argument('dataset', type=str, choices=['mnist'])
parser.add_argument('num_units', type=int)
parser.add_argument('--activation', choices=['sigmoid','linear','tanh'], default='sigmoid')
parser.add_argument('--loss',choices=[
    'mean_squared_error',
    'mean_absolute_error',
],default='mean_squared_error')
parser.add_argument('--bsize',type=int,default=64)

def main(args):

    dataset = datasets.Mnist()
    (x_train, t_train), (x_test, t_test) = dataset.load_data()
    for class_id in range(dataset.num_classes):
        # ========================================================
        # Prepare Dataset
        # ========================================================
        normal_train = x_train[t_train == class_id]
        normal_test = x_test[t_test == class_id]
        anomaly_test = x_test[t_test != class_id]
        border = int(1.2 * args.num_units)
        normal_ini_train = normal_train[:border]
        normal_seq_train = normal_train[border:]
        test = np.concatenate((normal_test, anomaly_test), axis=0)
        test_labels = np.concatenate(([0] * len(normal_test), [1] * len(anomaly_test)), axis=0)

        # ========================================================
        # Build model
        # ========================================================
        os_elm = OS_ELM(
            dataset.inputs,
            args.num_units,
            dataset.inputs,
            loss=args.loss,
            activation=args.activation,
        )

        # ========================================================
        # Initial training phase
        # ========================================================
        os_elm.init_train(normal_ini_train, normal_ini_train)

        # ========================================================
        # Sequential training phase
        # ========================================================
        for i in range(0, len(normal_seq_train), args.bsize):
            x = normal_seq_train[i : i + args.bsize]
            os_elm.seq_train(x, x)

        # ========================================================
        # Evaluation
        # ========================================================
        # calc score
        scores = []
        for i in range(len(test)):
            x = np.expand_dims(test[i], axis=0)
            score = os_elm.evaluate(x, x)
            scores.append(score)
        scores = np.array(scores)
        auc = roc_auc_score(test_labels, scores)
        print('%d: %f' % (class_id, auc))
        del os_elm

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
