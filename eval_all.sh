#!/bin/sh

for dataset in boston digits mnist
do
    command="sh eval_$dataset.sh"
    eval $command
done
