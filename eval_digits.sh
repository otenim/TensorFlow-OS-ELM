#!/bin/sh

for bsize in 8 16 32 64 128 256
do
    for units in 64 128 256 512 1024
    do
        command="python train_digits.py --batch_size=$bsize --units=$units"
        eval $command
    done
done
