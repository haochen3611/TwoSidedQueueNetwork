#!/bin/bash

FILE_DIR=$(dirname "$0")

python3 "${FILE_DIR}/run.py" \
  --num_cpus 100 \
  --num_gpus 0 \
  --iter 100 \
  --config "default.json"