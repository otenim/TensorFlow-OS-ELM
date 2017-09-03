import argparse
from keras.datasets import boston_housing
import numpy as np
import os
import time
from os_elm import OS_ELM
from utils import normalize_dataset

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
    default=64,
)
parser.add_argument(
    '--result_root',
    default=os.path.join(current_directory, 'result', 'boston'),
)

def main(args):

    # hyper parameters
    units = args.units
    batch_size = args.batch_size
    init_batch_size = int(args.units * 1.2)
    epochs = args.epochs
    result_root = os.path.abspath(args.result_root)
    inputs = 13
    outputs = 1
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
    (x_train, y_train), (x_test, y_test) = boston_housing.load_data()

    # normalize dataset into [0,1]
    x_train = normalize_dataset(x_train, axis=0)
    y_train = normalize_dataset(y_train, axis=0)
    x_test = normalize_dataset(x_test, axis=0)
    y_test = normalize_dataset(y_test, axis=0)

    # separate training data into two groups
    # (for initial training and for sequential training)
    x_train_init = x_train[:init_batch_size]
    y_train_init = y_train[:init_batch_size]
    x_train_seq = x_train[init_batch_size:]
    y_train_seq = y_train[init_batch_size:]

    # Experiment Loop
    sum_loss = 0.
    sum_val_loss = 0.
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

        # evaluation(training)
        loss = model.eval(x_train_seq[:n_test], y_train_seq[:n_test], mode='regress')
        print("training loss(MSE): %f" % (loss))
        sum_loss += loss

        # evaluation(validation)
        val_loss = model.eval(x_test, y_test, mode='regress')
        print("validation loss(MSE): %f" % (val_loss))
        sum_val_loss += val_loss

    # final evaluation
    print("/*========== FINAL RESULT ==========*/")
    print("mean single training time: %f[sec]" % (sum_single_training_time / epochs))
    print("mean training loss: %f" % (sum_loss / epochs))
    print("mean validation loss: %f" % (sum_val_loss / epochs))

    # write result
    if os.path.exists(result_root) == False:
        os.makedirs(result_root)
    fname = "batchsize%d_units%d.out" % (batch_size, units)
    with open(os.path.join(result_root, fname), 'w') as f:
        f.write("/*========== FINAL RESULT ==========*/\n")
        f.write("mean single training time: %f[sec]\n" % (sum_single_training_time / epochs))
        f.write("mean training loss: %f\n" % (sum_loss / epochs))
        f.write("mean validation loss: %f\n" % (sum_val_loss / epochs))
        f.write("/*========== HTML code for this record ==========*/\n")
        f.write("| units | time | val_loss |\n")
        f.write("<tr>\n")
        f.write("\t<td>%d</td>\n" % (units))
        f.write("\t<td>%f</td>\n" % (sum_single_training_time / epochs))
        f.write("\t<td>%f</td>\n" % (sum_val_loss / epochs))
        f.write("</tr>\n")


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
