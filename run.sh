#!/bin/bash

FILE_DIR=$(dirname "$0")

python3 "${FILE_DIR}/run.py" \
  --num_cpus 44 \
  --num_gpus 0 \
  --iter 1000 \
  --config "default.json"