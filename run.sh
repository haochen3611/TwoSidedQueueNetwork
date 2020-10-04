#!/bin/bash

FILE_DIR=$(dirname "$0")

python3 "${FILE_DIR}/run.py" --num_cpus 2 --num_gpus 1 --iter 10