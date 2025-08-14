#!/bin/bash
"""
Experiment runner script for comparing different generative models.
This script runs experiments with different model configurations.
"""

# Script to run comparison experiments

echo "Starting model comparison experiments..."

# Base command
BASE_CMD="python train_comparison.py"

# Define experiments
declare -a models=("wgan" "vae" "standard_gan")

# Run experiments for each model
for model in "${models[@]}"; do
    echo "Running experiment with model: $model"
    
    # Create experiment-specific command
    CMD="$BASE_CMD model=$model train=$model inference=$model experiment.name=comparison_$model"
    
    # Add debug flag if needed
    if [[ "$1" == "debug" ]]; then
        CMD="$CMD debug=True"
    fi
    
    echo "Executing: $CMD"
    eval $CMD
    
    echo "Completed experiment with model: $model"
    echo "-----------------------------------"
done

echo "All experiments completed!"
