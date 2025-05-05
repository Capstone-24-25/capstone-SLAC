#!/bin/bash
export CUDA_VISIBLE_DEVICES=$1
python sample_benchmark.py --method stratified --encoded_dim 2048 --ae_epochs 20 --num_epochs 10 --batch_size 5