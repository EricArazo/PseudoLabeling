#!/bin/bash

# Warm up of  10 epochs
python3 train.py --labeled_samples 4000 --epoch 10 --dataset_type "ssl_warmUp" \
--DA "jitter" --experiment_name "WuP_model" --download "True"

# SSL training
python3 train.py --labeled_samples 4000 --epoch 400  --M 250 --M 350 --load_epoch 10 \
--DA "jitter" --experiment_name "M_SOTA_MINIIMAGENET" --download "True"
