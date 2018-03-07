# OS-ELM-with-Python

## Overview

<img src="https://i.imgur.com/GckJu86.png" width=500>

In this repository, we provide an implementation of Online Sequential
Extreme Machine (OS-ELM) introduced by Liang et al. in 2006.
OS-ELM is known to be able to trian faster and always converge to the global optimal solution.
Also, since OS-ELM has smaller number of hyperparameters than backpropagation-based
neural networks, it can provide a high generalization performance.

[Paper](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.217.1418&rep=rep1&type=pdf).

## Requsite Libraries

We tested our codes using the following libraries.

* Python==3.6.0
* Numpy==1.14.1
* Keras==2.1.4
* scikit-learn==0.17.1

You don't have to use the exactly same version of the each library,
but we can't guarantee codes work well in that case.

All requsite libraries above can be installed in the following command.

`$ pip install -U numpy Keras scikit-learn`

## Usage

Here, we describe how to train OS-ELM and predict with the trained model.  
For the sake of simplicity, we assume to train a model on 'MNIST' dataset.  
The following example shows a standard code pipeline for the use case.

```python
from keras.datasets import mnist
from keras.utils import to_categorical
from os_elm import OS_ELM, load_model
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--n_hidden_nodes', type=int, default=256)
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
        # number of nodes of input layer
        n_input_nodes=n_input_dimensions,
        # number of nodes of hidden layer
        n_hidden_nodes=args.n_hidden_nodes,
        # number of nodes of output layer
        n_output_nodes=n_classes, # 10
        # activation function
        # support 'sigmoid' and 'linear' so far.
        # the default value is 'sigmoid'.
        activation=args.activation,
        # loss function
        # support 'mean_squared_error' and 'mean_absolute_error' so far.
        # the default value is 'mean_squared_error'.
        loss=args.loss
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
    os_elm.init_train(
        x_train_init,
        y_train_init
    )

    # sequential training phase
    os_elm.seq_train(
        x_train_seq,
        y_train_seq,
        # batch size during sequential training phase
        # the default value is 32.
        batch_size=args.batch_size,
        # whether to show a progress bar or not
        # the default value is 1.
        verbose=1,
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
    n = 5
    x = x_test[:n]
    y_pred = os_elm.predict(x, softmax=True)

    for i in range(n):
        print("========== Prediction result (%d) ==========" % i)
        class_id = np.argmax(y_pred[i])
        print("class_id (prediction): %d" % class_id)
        print("class_id (true): %d" % np.argmax(y_test[i]))
        class_prob = y_pred[i][class_id]
        print("probability (prediction): %.3f" % class_prob)

    # ===========================================
    # Evaluation
    # ===========================================
    print("========== Evaluation result ==========")
    loss, acc = os_elm.evaluate(
        x_test,
        y_test,
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

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
```

## Notes

The following figure shows OS-ELM training formula.  

<img src="https://i.imgur.com/QjqaMcS.png" width=500>

* OS-ELM always converge to the global optimal solutions, while backpropagation-based neural networks
tend to suffer from local minima problem.
* OS-ELM does not need to train iteratively on the same data samples.
Even if you did it, the computational result will not change at all.
* OS-ELM does not need to compute gradients. The weights are trained by
computing some matrix multipies and a matrix inversion.
* The computational complexity for the matrix inversion is O(batch_size^3),
so take care for the cost when you increase batch_size.

## Demo

The above code is summarized in train_mnist.py. Here, we describe how to
execute the script.

`$ python train_mnist.py [--n_hidden_nodes] [--batch_size] [--activation] [--loss]`

* [optional] `--n_hidden_nodes`: number of nodes of hidden layer.
the default value is set to 512.
* [optional] `--batch_size`: batch size during sequential training phase.
the default value is set to 32.
* [optional] `--activation`: activation function to be applied in hidden nodes.
we support 'sigmoid' and 'linear' so far. the default value is set to 'sigmoid'.
* [optional] `--loss`: loss function to be applied in model's output.
we support 'mean_squared_error' and 'mean_absolute_error' so far.
the default value is set to 'mean_squared_error'.

The following command is an example.

`$ python train.py --n_hidden_nodes 512 --batch_size 32 --activation sigmoid --loss mean_squared_error`

## Todos

* support more activation functions
* support more loss functions
* provide GPU implementation using CuPy
* provide benchmark results
