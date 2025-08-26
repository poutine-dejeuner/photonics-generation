#!/bin/bash
models=("wgan" "standard_gan" "vae" "simple_unet")

for (( a = 0; a < 4; a++ ))
do
model=${models[$a]}
python train_comparison.py model=$model train=$model logger=True debug=True
done
