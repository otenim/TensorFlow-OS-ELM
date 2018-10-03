from keras.datasets import mnist
from keras.utils import to_categorical
from os_elm import OS_ELM
import numpy as np
import tensorflow as tf
import tqdm

def softmax(a):
    c = np.max(a, axis=-1).reshape(-1, 1)
    exp_a = np.exp(a - c)
    sum_exp_a = np.sum(exp_a, axis=-1).reshape(-1, 1)
    return exp_a / sum_exp_a

def main():

    # ===========================================
    # Instantiate os-elm
    # ===========================================
    n_input_nodes = 784
    n_hidden_nodes = 1024
    n_output_nodes = 10

    os_elm = OS_ELM(
        # the number of input nodes.
        n_input_nodes=n_input_nodes,
        # the number of hidden nodes.
        n_hidden_nodes=n_hidden_nodes,
        # the number of output nodes.
        n_output_nodes=n_output_nodes,
        # loss function.
        # the default value is 'mean_squared_error'.
        # for the other functions, we support
        # 'mean_absolute_error', 'categorical_crossentropy', and 'binary_crossentropy'.
        loss='mean_squared_error',
        # activation function applied to the hidden nodes.
        # the default value is 'sigmoid'.
        # for the other functions, we support 'linear' and 'tanh'.
        # NOTE: OS-ELM can apply an activation function only to the hidden nodes.
        activation='sigmoid',
    )

    # ===========================================
    # Prepare dataset
    # ===========================================
    n_classes = n_output_nodes

    # load MNIST
    (x_train, t_train), (x_test, t_test) = mnist.load_data()
    # normalize images' values within [0, 1]
    x_train = x_train.reshape(-1, n_input_nodes) / 255.
    x_test = x_test.reshape(-1, n_input_nodes) / 255.
    x_train = x_train.astype(np.float32)
    x_test = x_test.astype(np.float32)

    # convert label data into one-hot-vector format data.
    t_train = to_categorical(t_train, num_classes=n_classes)
    t_test = to_categorical(t_test, num_classes=n_classes)
    t_train = t_train.astype(np.float32)
    t_test = t_test.astype(np.float32)

    # divide the training dataset into two datasets:
    # (1) for the initial training phase
    # (2) for the sequential training phase
    # NOTE: the number of training samples for the initial training phase
    # must be much greater than the number of the model's hidden nodes.
    # here, we assign int(1.5 * n_hidden_nodes) training samples
    # for the initial training phase.
    border = int(1.5 * n_hidden_nodes)
    x_train_init = x_train[:border]
    x_train_seq = x_train[border:]
    t_train_init = t_train[:border]
    t_train_seq = t_train[border:]


    # ===========================================
    # Training
    # ===========================================
    # the initial training phase
    pbar = tqdm.tqdm(total=len(x_train), desc='initial training phase')
    os_elm.init_train(x_train_init, t_train_init)
    pbar.update(n=len(x_train_init))

    # the sequential training phase
    pbar.set_description('sequential training phase')
    batch_size = 64
    for i in range(0, len(x_train_seq), batch_size):
        x_batch = x_train_seq[i:i+batch_size]
        t_batch = t_train_seq[i:i+batch_size]
        os_elm.seq_train(x_batch, t_batch)
        pbar.update(n=len(x_batch))
    pbar.close()

    # ===========================================
    # Prediction
    # ===========================================
    # sample 10 validation samples from x_test
    n = 10
    x = x_test[:n]
    t = t_test[:n]

    # 'predict' method returns raw values of output nodes.
    y = os_elm.predict(x)
    # apply softmax function to the output values.
    y = softmax(y)
    
    # check the answers.
    for i in range(n):
        max_ind = np.argmax(y[i])
        print('========== sample index %d ==========' % i)
        print('estimated answer: class %d' % max_ind)
        print('estimated probability: %.3f' % y[i,max_ind])
        print('true answer: class %d' % np.argmax(t[i]))

    # ===========================================
    # Evaluation
    # ===========================================
    # we currently support 'loss' and 'accuracy' for 'metrics'.
    # NOTE: 'accuracy' is valid only if the model assumes
    # to deal with a classification problem, while 'loss' is always valid.
    # loss = os_elm.evaluate(x_test, t_test, metrics=['loss']
    [loss, accuracy] = os_elm.evaluate(x_test, t_test, metrics=['loss', 'accuracy'])
    print('val_loss: %f, val_accuracy: %f' % (loss, accuracy))

    # ===========================================
    # Save model
    # ===========================================
    print('saving model parameters...')
    os_elm.save('./checkpoint/model.ckpt')

    # initialize weights of os_elm
    os_elm.initialize_variables()

    # ===========================================
    # Load model
    # ===========================================
    # If you want to load weights to a model,
    # the architecture of the model must be exactly the same
    # as the one when the weights were saved.
    print('restoring model parameters...')
    os_elm.restore('./checkpoint/model.ckpt')

    # ===========================================
    # ReEvaluation
    # ===========================================
    # loss = os_elm.evaluate(x_test, t_test, metrics=['loss']
    [loss, accuracy] = os_elm.evaluate(x_test, t_test, metrics=['loss', 'accuracy'])
    print('val_loss: %f, val_accuracy: %f' % (loss, accuracy))

if __name__ == '__main__':
    main()
