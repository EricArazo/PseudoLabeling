#!/bin/bash

# Warm up of  10 epochs
python3 train.py --labeled_samples 500 --epoch 10 --dataset_type "ssl_warmUp" \
--dropout 0.1 --DA "jitter" --experiment_name "WuP_model" --download "True"

# SSL training
python3 train.py --labeled_samples 500 --epoch 400  --M 250 --M 350 --load_epoch 10 \
--dropout 0.1 --DA "jitter" --experiment_name "M_SOTA_CIFAR10" --download "True"
