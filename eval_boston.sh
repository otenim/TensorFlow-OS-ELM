#!/bin/sh

for bsize in 4 8 16 32 64
do
    for units in 8 16 32 64 128
    do
        command="python train_boston.py --batch_size=$bsize --units=$units"
        eval $command
    done
done
