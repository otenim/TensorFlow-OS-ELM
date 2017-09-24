import argparse
from sklearn.datasets import load_digits
import numpy as np
import os
import time
from os_elm import OS_ELM
from utils import normalize_dataset
from utils import dump_dataset, dump_matrix

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
    default=50,
)
parser.add_argument(
    '--units',
    type=int,
    default=128,
)
parser.add_argument(
    '--actfun',
    choices=['sigmoid', 'relu'],
    default='sigmoid',
)
parser.add_argument(
    '--result_root',
    default=os.path.join(current_directory, 'result', 'digits'),
)
parser.add_argument(
    '--save_weights',
    type=bool,
    default=False,
)
parser.add_argument(
    '--dump_dataset',
    type=bool,
    default=False,
)
parser.add_argument(
    '--dataset_root',
    default=os.path.join(current_directory, 'datasets', 'digits')
)
parser.add_argument(
    '--weights_root',
    default=os.path.join(current_directory, 'weights', 'digits'),
)

def main(args):

    # hyper parameters
    units = args.units
    batch_size = args.batch_size
    init_batch_size = int(args.units * 1.2)
    epochs = args.epochs
    result_root = os.path.abspath(args.result_root)
    weights_root = os.path.abspath(args.weights_root)
    actfun = args.actfun
    inputs = 64
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
    x, y = load_digits().data, load_digits().target
    split = int(len(x) * 0.8)
    x_train, x_test = x[:split], x[split:]
    y_train, y_test = y[:split], y[split:]

    # normalize input dataset into [0,1.0]
    x_train = normalize_dataset(x_train, axis=None)
    x_test = normalize_dataset(x_test, axis=None)

    # translate labels into one-hot vectors
    y_test = np.eye(outputs)[y_test]
    y_train = np.eye(outputs)[y_train]

    # dump dataset
    if args.dump_dataset:
        root = os.path.abspath(args.dataset_root)
        dump_matrix(os.path.join(root, 'x_train_digits.dat'), x_train)
        dump_matrix(os.path.join(root, 'x_test_digits.dat'), x_test)
        dump_matrix(os.path.join(root, 'y_train_digits.dat'), y_train)
        dump_matrix(os.path.join(root, 'y_test_digits.dat'), y_test)

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
        model = OS_ELM(inputs=inputs, units=units, outputs=outputs, activation=actfun)

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
        f.write("/*========== HTML code for this record ==========*/\n")
        f.write("| units | time | val_loss | val_acc |\n")
        f.write("<tr>\n")
        f.write("\t<td>%d</td>\n" % (units))
        f.write("\t<td>%f</td>\n" % (sum_single_training_time / epochs))
        f.write("\t<td>%f</td>\n" % (sum_val_loss / epochs))
        f.write("\t<td>%f</td>\n" % (sum_val_acc / epochs))
        f.write("</tr>\n")

    # save weights
    if args.save_weights:
        if os.path.exists(weights_root) == False:
            os.makedirs(weights_root)
        dump_matrix(os.path.join(weights_root, 'batchsize%d_units%d.beta' % (batch_size, units)), model.beta)
        dump_matrix(os.path.join(weights_root, 'batchsize%d_units%d.beta_init' % (batch_size, units)), model.beta_init)
        dump_matrix(os.path.join(weights_root, 'batchsize%d_units%d.beta_rand' % (batch_size, units)), model.beta_rand)
        dump_matrix(os.path.join(weights_root, 'batchsize%d_units%d.alpha' % (batch_size, units)), model.alpha)
        dump_matrix(os.path.join(weights_root, 'batchsize%d_units%d.alpha_rand' % (batch_size, units)), model.alpha_rand)
        dump_matrix(os.path.join(weights_root, 'batchsize%d_units%d.p' % (batch_size, units)), model.p)
        dump_matrix(os.path.join(weights_root, 'batchsize%d_units%d.p_init' % (batch_size, units)), model.p_init)



if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
