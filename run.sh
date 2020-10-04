#!/bin/bash

FILE_DIR=$(readlink -f "$0")

python3 "$FILE_DIR" --num_cpus 2 --num_gpus 1 --iter 10