import numpy as np
import argparse
import os
import tqdm
import datasets
import time
import random
from nn import NN
from sklearn.metrics import roc_auc_score

parser = argparse.ArgumentParser()
parser.add_argument('dataset', type=str, choices=['mnist'])
parser.add_argument('num_units', type=int)
parser.add_argument('--unit_activation', choices=['sigmoid', 'linear', 'relu'], default='relu')
parser.add_argument('--out_activation', choices=['sigmoid', 'linear', 'relu'], default='sigmoid')
parser.add_argument('--loss', choices=[
    'mean_squared_error',
    'mean_absolute_error',
],default='mean_squared_error')
parser.add_argument('--opt', choices=['adam', 'sgd'], default='adam')
parser.add_argument('--bsize', type=int, default=64)
parser.add_argument('--epochs', type=int, default=10)

def main(args):

    dataset = datasets.Mnist()
    (x_train, t_train), (x_test, t_test) = dataset.load_data()

    # ========================================================
    # Build model
    # ========================================================
    model = NN(
        dataset.inputs,
        args.num_units,
        dataset.inputs,
        loss=args.loss,
        hidden_activation=args.unit_activation,
        output_activation=args.out_activation,
        optimizer=args.opt,
    )

    for class_id in range(dataset.num_classes):
        # ========================================================
        # Prepare Dataset
        # ========================================================
        normal_train = x_train[t_train == class_id]
        normal_test = x_test[t_test == class_id]
        anomaly_test = x_test[t_test != class_id]
        test = np.concatenate((normal_test, anomaly_test), axis=0)
        test_labels = np.concatenate(([0] * len(normal_test), [1] * len(anomaly_test)), axis=0)

        # ========================================================
        # Training
        # ========================================================
        for _ in range(args.epochs):
            for i in range(0, len(normal_train), args.bsize):
                x = normal_train[i : i + args.bsize]
                model.train_on_batch(x, x)

        # ========================================================
        # Evaluation
        # ========================================================
        # calc score
        scores = []
        for i in range(len(test)):
            x = np.expand_dims(test[i], axis=0)
            score = model.test_on_batch(x, x)[0]
            scores.append(score)
        scores = np.array(scores)
        auc = roc_auc_score(test_labels, scores)
        print('%d: %f' % (class_id, auc))
        
        # reset
        model.initialize_variables()

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
