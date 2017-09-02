#!/bin/sh

for bsize in 32 64 128 256 512
do
    for units in 128 256 512 1024
    do
        command="python train_mnist.py --batch_size=$bsize --units=$units"
        eval $command
    done
done
