#!/bin/bash
# Running things

# Warm up of  10 epochs
python3 train.py --labeled_samples 1000 --epoch 10 --label_noise 0.0 --dataset_type "sym_noise_warmUp" --experiment_name "WuP_model"

# SSL training
python3 train.py --labeled_samples 1000 --epoch 400  --M 250 --M 350 --label_noise 0.2 --initial_epoch 10 --experiment_name "M_02LN"
