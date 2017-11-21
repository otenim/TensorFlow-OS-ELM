#!/bin/sh

activation="sigmoid"
loss="mean_squared_error"
dataset="mnist"
result_dir="./results"

for batch_size in 4 8 16 32 64
do
    for units in 256 512 1024 2048
    do
        python train.py --dataset $dataset --units $units --batch_size $batch_size --activation $activation --loss $loss --result $result_dir/$dataset
    done
done
