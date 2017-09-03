import argparse
import chainer
import numpy as np
import os
import time
from os_elm import OS_ELM

current_directory = os.path.dirname(os.path.abspath(__file__))
parser = argparse.ArgumentParser()
parser.add_argument(
    '--batch_size',
    type=int,
    default=32,
)
parser.add_argument(
    '--epochs',
    type=int,
    default=20,
)
parser.add_argument(
    '--units',
    type=int,
    default=128,
)
parser.add_argument(
    '--result_root',
    default=os.path.join(current_directory, 'result', 'mnist'),
)

def main(args):

    # hyper parameters
    units = args.units
    batch_size = args.batch_size
    init_batch_size = int(args.units * 1.2)
    epochs = args.epochs
    result_root = os.path.abspath(args.result_root)
    inputs = 784
    outputs = 10
    print("/*========== Info ==========*/")
    print("inputs: %d" % inputs)
    print("units: %d" % units)
    print("outputs: %d" % outputs)
    print("batch_size: %d" % batch_size)
    print("init_batch_size: %d" % init_batch_size)
    print("epochs: %d" % epochs)
    print("result_root: %s" % result_root)

    # prepare datasets
    print("preparing dataset...")
    train, test = chainer.datasets.get_mnist()
    x_train, y_train = train._datasets[0], train._datasets[1]
    x_test, y_test = test._datasets[0], test._datasets[1]

    # translate labels into one-hot vectors
    y_test = np.eye(outputs)[y_test]
    y_train = np.eye(outputs)[y_train]

    # separate training data into two groups
    # (for initial training and for sequential training)
    x_train_init = x_train[:init_batch_size]
    y_train_init = y_train[:init_batch_size]
    x_train_seq = x_train[init_batch_size:]
    y_train_seq = y_train[init_batch_size:]

    # Experiment Loop
    sum_loss, sum_acc = 0., 0.
    sum_val_loss, sum_val_acc = 0., 0.
    sum_single_training_time = 0.

    for epoch in range(epochs):
        print("/*========== TRAINING EPOCH %d ==========*/" % (epoch + 1))
        # instantiate model
        print("creating model...")
        model = OS_ELM(inputs=inputs, units=units, outputs=outputs)

        # ===================================
        # initial training phase
        # ===================================
        print("initial training phase...")
        model.init_train(x_train_init, y_train_init)

        # ===================================
        # sequential training phase
        # ===================================
        print("sequential training phase...")
        n_train_seq = len(x_train_seq)
        n_test = len(x_test)

        # shuffle data
        idx = np.random.permutation(n_train_seq)
        x_train_seq = x_train_seq[idx]
        y_train_seq = y_train_seq[idx]

        # train model on given dataset
        single_training_time = 0.
        cnt = 0
        for i in range(0, n_train_seq, batch_size):
            x, y = x_train_seq[i:i+batch_size], y_train_seq[i:i+batch_size]
            start = time.time()
            model.seq_train(x, y)
            single_training_time += (time.time() - start)
            cnt += 1
        sum_single_training_time += (single_training_time / cnt)
        print("single training time: %f[sec]" % (single_training_time / cnt))

        # evaluation
        loss, acc = model.eval(x_train_seq[:n_test], y_train_seq[:n_test])
        print("training loss(MSE): %f" % (loss))
        print("training accuracy: %f%%" % (acc))
        sum_loss += loss
        sum_acc += acc

        val_loss, val_acc = model.eval(x_test, y_test)
        print("validation loss(MSE): %f" % (val_loss))
        print("validation accuracy: %f%%" % (val_acc))
        sum_val_loss += val_loss
        sum_val_acc += val_acc

    # final evaluation
    print("/*========== FINAL RESULT ==========*/")
    print("mean single training time: %f[sec]" % (sum_single_training_time / epochs))
    print("mean training loss(MSE): %f" % (sum_loss / epochs))
    print("mean training accuracy: %f%%" % (sum_acc / epochs))
    print("mean validation loss(MSE): %f" % (sum_val_loss / epochs))
    print("mean validation accuracy: %f%%" % (sum_val_acc / epochs))

    # write result
    if os.path.exists(result_root) == False:
        os.makedirs(result_root)
    fname = "batchsize%d_units%d.out" % (batch_size, units)
    with open(os.path.join(result_root, fname), 'w') as f:
        f.write("/*========== FINAL RESULT ==========*/\n")
        f.write("mean single training time: %f[sec]\n" % (sum_single_training_time / epochs))
        f.write("mean training loss(MSE): %f\n" % (sum_loss / epochs))
        f.write("mean training accuracy: %f%%\n" % (sum_acc / epochs))
        f.write("mean validation loss(MSE): %f\n" % (sum_val_loss / epochs))
        f.write("mean validation accuracy: %f%%\n" % (sum_val_acc / epochs))


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
