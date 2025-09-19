#!/bin/bash
# models=("wgan" "standard_gan" "vae" "simple_unet")
models=("unet_fast")
first_channel=(4 8)
# first_channel=(4 8 16 32)

for model in "${models[@]}"; do
  for d in "${first_channel[@]}"; do
  echo $model
  python photo_gen/train_comparison.py model=$model train=$model logger=False debug=True model.first_channels=$d
  done
done
