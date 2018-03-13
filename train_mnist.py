from keras.datasets import mnist
from keras.utils import to_categorical
from os_elm import OS_ELM
import numpy as np
import argparse
import tensorflow as tf
import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--n_hidden_nodes', type=int, default=1024)
parser.add_argument('--batch_size', type=int, default=32)

def main(args):

    # ===========================================
    # Prepare dataset
    # ===========================================
    n_input_dimensions = 784
    n_classes = 10

    (x_train, t_train), (x_test, t_test) = mnist.load_data()
    x_train = x_train.reshape(-1, n_input_dimensions) / 255.
    x_test = x_test.reshape(-1, n_input_dimensions) / 255.
    x_train = x_train.astype(np.float32)
    x_test = x_test.astype(np.float32)

    t_train = to_categorical(t_train, num_classes=n_classes)
    t_test = to_categorical(t_test, num_classes=n_classes)
    t_train = t_train.astype(np.float32)
    t_test = t_test.astype(np.float32)

    border = int(1.5 * args.n_hidden_nodes)
    x_train_init = x_train[:border]
    x_train_seq = x_train[border:]
    t_train_init = t_train[:border]
    t_train_seq = t_train[border:]


    # ===========================================
    # Instantiate os-elm
    # ===========================================
    os_elm = OS_ELM(
        # number of nodes of input layer
        n_input_nodes=n_input_dimensions,
        # number of nodes of hidden layer
        n_hidden_nodes=args.n_hidden_nodes,
        # number of nodes of output layer
        n_output_nodes=n_classes, # 10
    )


    # ===========================================
    # training
    # ===========================================
    # initial training phase
    pbar = tqdm.tqdm(total=len(x_train), desc='initial training phase')
    os_elm.init_train(x_train_init, t_train_init)
    pbar.update(n=len(x_train_init))

    # sequential training phase
    pbar.set_description('sequential training phase')
    for i in range(0, len(x_train_seq), args.batch_size):
        x_batch = x_train_seq[i:i+args.batch_size]
        t_batch = t_train_seq[i:i+args.batch_size]
        os_elm.seq_train(x_batch, t_batch)
        pbar.update(n=len(x_batch))
    pbar.close()

    [loss, accuracy] = os_elm.evaluate(x_test, t_test, metrics=['loss', 'accuracy'])
    print('val_loss: %f, val_accuracy: %f' % (loss, accuracy))

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
