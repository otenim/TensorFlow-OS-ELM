#!/bin/sh

activation="sigmoid"
loss="mean_squared_error"
dataset="digits"
result_dir="./results"

for batch_size in 4 8 16 32 64
do
    for units in 32 64 128 256 512
    do
        python train.py --dataset $dataset --units $units --batch_size $batch_size --activation $activation --loss $loss --result $result_dir/$dataset
    done
done
