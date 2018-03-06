from keras.datasets import mnist
from keras.utils import to_categorical
from os_elm import OS_ELM
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--n_hidden_nodes', type=int, default=1024)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--activation',
    choices=['sigmoid','linear'], default='sigmoid')
parser.add_argument('--loss',
    choices=['mean_squared_error', 'mean_absolute_error'], default='mean_squared_error')

def main(args):

    # ===========================================
    # Prepare dataset
    # ===========================================
    n_input_dimensions = 784
    n_classes = 10

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    # normalize pixel values of input images within [0,1]
    x_train = x_train.astype(np.float32) / 255.
    x_test = x_test.astype(np.float32) / 255.
    # reshape
    x_train = x_train.reshape(-1,n_input_dimensions)
    x_test = x_test.reshape(-1,n_input_dimensions)
    # transform label data in one-hot-vector format
    y_train = to_categorical(y_train, num_classes=n_classes)
    y_test = to_categorical(y_test, num_classes=n_classes)

    # ===========================================
    # Instantiate os-elm
    # ===========================================
    os_elm = OS_ELM(
        n_input_nodes=n_input_dimensions, # 784
        n_hidden_nodes=args.n_hidden_nodes, # user define
        n_output_nodes=n_classes, # 10
        activation=args.activation, # support 'sigmoid' and 'linear' so far
        loss=args.loss, # support 'mean_squared_error' and 'mean_absolute_error' so far
    )


    # ===========================================
    # training
    # ===========================================
    # devide x_train into two dataset.
    # the one is for initial training phase,
    # the other is for sequential training phase.
    # NOTE: number of data samples for initial training phase
    # should be much greater than os_elm.n_hidden_nodes.
    # here, we set 1.1 * os_elm.n_hidden_nodes as the
    # number of data samples for initial training phase.
    border = int(1.1 * os_elm.n_hidden_nodes)
    x_train_init = x_train[:border]
    y_train_init = y_train[:border]
    x_train_seq = x_train[border:]
    y_train_seq = y_train[border:]

    # initial training phase
    os_elm.init_train(x_train_init, y_train_init)

    # sequential training phase
    os_elm.seq_train(
        x_train_seq,
        y_train_seq,
        batch_size=args.batch_size, # batch size during training
        verbose=1, # whether to show a progress bar or not
    )

    # ===========================================
    # Prediction
    # ===========================================
    # NOTE: input numpy arrays' shape is always assumed to
    # be in the following 2D format: (batch_size, n_input_nodes).
    # Even when you feed one training sample to the model,
    # the input sample's shape must be (1, n_input_nodes), not
    # (n_input_nodes,). Here, we feed one validation sample
    # as an example.
    x = x_test[0]
    x = np.expand_dims(x, axis=0)
    y_pred = os_elm.predict(x, softmax=True)
    class_id = np.argmax(y_pred[0])
    class_prob = y_pred[0][class_id]
    print("class_id (prediction): %d" % class_id)
    print("probability (prediction): %.3f" % class_prob)
    print("class_id (true): %d" % np.argmax(y_test[0]))

    # ===========================================
    # Evaluation
    # ===========================================
    loss, acc = os_elm.evaluate(x_test, y_test, metrics=['loss', 'accuracy'])
    print('validation loss: %f' % loss)
    print('classification accuracy: %f' % acc)

    # ===========================================
    # Save model
    # ===========================================
    os_elm.save('model.pkl')

    # ===========================================
    # Load model
    # ===========================================

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
