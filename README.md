# OS-ELM-with-Python

## Overview

In this repository, we produce an implementation of Online Sequential
Extreme Machin (OS-ELM) introduced by Liang et al. in 2006.
OS-ELM is known to be able to train faster and more accurately than
the other sequential learning algorithms including
backpropagation-based neural networks.

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

We describe how to train OS-ELM and predict with it. For the sake of simplicity, we assume to use 'Mnist' as a dataset here.

### 1. Instantiate model

```python
from models import OS_ELM

INPUT_NODES=784 # Number of input nodes
HIDDEN_NODES=1024 # Number of hidden nodes
OUTPUT_NODES=10 # Number of output nodes
BATCH_SIZE=128 # Batch size NOTE: not have to be fixed-length

os_elm = OS_ELM(
    inputs=INPUT_NODES,
    units=HIDDEN_NODES,
    outputs=OUTPUT_NODES,
    activation='sigmoid', # 'sigmoid' or 'relu'
    loss='mean_squared_error' # we support 'mean_squared_error' only
)
```

### 2. Prepare dataset

```python
from datasets import Mnist

# x_train: shape=(60000,784), dtype=float32, normalized in [0,1]
# y_train: shape=(60000,10), dtype=float32, one-hot-vector format
# x_test: shape=(10000,784), dtype=float32, normalized in [0,1]
# y_test: shape=(10000,10), dtype=float32, one-hot-vector format

dataset = Mnist()
(x_train, y_train), (x_test, y_test) = dataset.load_data()

# Separate training dataset for initial training phase
# and sequential training phase.
# NOTE: batch size of initial training dataset must be
# much greater than the number of hidden units of os_elm.

border = int(HIDDEN_NODES * 1.1)
x_train_init, x_train_seq = x_train[:border], x_train[border:]
y_train_init, y_train_seq = y_train[:border], y_train[border:]
```

### 3. Training

```python
# Initial training phase
os_elm.init_train(x_train_init, y_train_init)

# Sequential training phase
for i in range(0, len(x_train_seq), BATCH_SIZE):
    x = x_train_seq[i:i+BATCH_SIZE]
    y = y_train_seq[i:i+BATCH_SIZE]
    os_elm.seq_train(x,y)
```

### 4. Predict

```python

# 'forward' method just forward the input data and return the outputs
# This method can be used for any type of problems.
# out.shape = (10000,10)
out = os_elm.forward(x_test)

# 'classify' method forward the input data and return the probability
# for each class.
# NOTE: This method can only be used for classification problems.
# out.shape = (10000,10)
prob = os_elm.classify(x_test)
```

### 5. Evaluation

```python

# Compute loss
# 'compute_loss' method can be used for any problems.
loss = os_elm.compute_loss(x_test,y_test)

# Compute accuracy
# NOTE: 'compute_accuracy' method can only be used for classification problems
acc = os_elm.compute_accuracy(x_test,y_test)

print('test_loss: %f' % (loss))
print('test_accuracy: %f' % (acc))
```

Above work processes are summarized in `sample.py`.  
It can be executed with the following command.  
`$ python sample.py`

## DEMO

`$ python train.py [--result] [--dataset] [--units] [--batch_size] [--activation] [--loss]`  

* `--result`: path to the directory which saves results.
* `--dataset`: 'mnist' or 'fashion' or 'digits' or 'boston'
    * 'fashion' means fashion\_mnist, and 'digits' is a small-size version mnist. 'boston' means boston\_housing dataset.
* `--units`: number of hidden units.
* `--batch_size`: mini-batch size.
* `--activation`: activation function to compute hidden nodes. we only support 'sigmoid' for now.
* `--loss`: loss function to compute error. we only support 'mean\_squared\_error' for now.

Following command is an example.  
`$ python train.py --result ./result --dataset mnist --units 1024 --batch_size 32 --activation sigmoid --loss mean_squared_error`

## Experimental results

see our [wiki](https://github.com/otenim/OS-ELM-with-Python/wiki)
