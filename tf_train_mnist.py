from keras.datasets import mnist
from keras.utils import to_categorical
from tf_os_elm import OS_ELM
import numpy as np
import argparse
import tensorflow as tf

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
    os_elm.fit(x_train, t_train, batch_size=args.batch_size, verbose=1)

    """
    # ===========================================
    # Prediction
    # ===========================================
    # NOTE: input numpy arrays' shape is always assumed to
    # be in the following 2D format: (batch_size, n_input_nodes).
    # Even when you feed one training sample to the model,
    # the input sample's shape must be (1, n_input_nodes), not
    # (n_input_nodes,). Here, we feed one validation sample
    # as an example.
    n = 5
    x = x_test[:n]
    y_pred = os_elm.predict(x, softmax=True)

    for i in range(n):
        print("========== Prediction result (%d) ==========" % i)
        class_id = np.argmax(y_pred[i])
        print("class_id (prediction): %d" % class_id)
        print("class_id (true): %d" % np.argmax(t_test[i]))
        class_prob = y_pred[i][class_id]
        print("probability (prediction): %.3f" % class_prob)

    # ===========================================
    # Evaluation
    # ===========================================
    print("========== Evaluation result ==========")
    loss, acc = os_elm.evaluate(
        x_test,
        t_test,
        # 'loss' and 'accuracy' are supported so far.
        # the default value is ['loss']
        # NOTE: 'accuracy' is only available for classification problems.
        metrics=['loss', 'accuracy']
    )
    print('validation loss: %f' % loss)
    print('classification accuracy: %f' % acc)

    # ===========================================
    # Save model
    # ===========================================
    print("saving trained model as model.pkl...")
    os_elm.save('model.pkl')

    # ===========================================
    # Load model
    # ===========================================
    print('loading model froom model.pkl...')
    os_elm = load_model('model.pkl')
    """

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
