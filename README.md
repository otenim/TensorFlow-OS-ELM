# OS-ELM-with-Python

## Overview

This repository produces an implementation of Online Sequential Extreme Learning Machine(OS-ELM) introduced in this [paper](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.217.1418&rep=rep1&type=pdf).

## Dependencies

* Python==3.5.2
* Numpy==1.13.3
* Keras==2.1.1
* scikit-learn==0.17.1

## DEMO

`$ python train.py [--dataset] [--units] [--batch_size] [--activation] [--loss]`  

* `--dataset`: 'mnist' or 'fashion' or 'digits' or 'boston'
    * 'fashion' means fashion_mnist, and 'digits' is a small-size version mnist. 'boston' means boston_housing dataset.
* `--units`: number of hidden units.
* `--batch_size`: mini-batch size.
* `--activation`: activation function to compute hidden nodes. we only support 'sigmoid' for now.
* `--loss`: loss function to compute error. we only support 'mean_squared_error' for now.

Following command is an example.  
`$ python train.py --dataset mnist --units 1024 --batch_size 32 --activation sigmoid --loss mean_squared_error`

## Experimental results

see our [wiki](https://github.com/otenim/OS-ELM-with-Python/wiki)
