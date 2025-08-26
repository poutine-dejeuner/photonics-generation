#!/bin/bash

# SLURM array job script for comparing different generative models
# This script runs experiments with different model configurations using SLURM array jobs

# Array of models to compare
models=("wgan" "vae" "standard_gan" "simple_unet")

# Create slurm directory if it doesn't exist
mkdir -p slurm

# Calculate total number of jobs
total_jobs=${#models[@]}
max_array_index=$((total_jobs - 1))

echo "Submitting array job for model comparison experiments..."
echo "Models: ${models[@]}"
echo "Total jobs: ${total_jobs} (array indices 0-${max_array_index})"

# Check if debug mode is requested
debug_flag=""
if [[ "$1" == "debug" ]]; then
    debug_flag="debug=True"
    echo "Running in debug mode"
fi

sbatch << EOF
#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 24:00:00
#SBATCH --array=0-${max_array_index}
#SBATCH --output slurm/comparison_%A_%a.out
#SBATCH --error slurm/comparison_%A_%a.err
#SBATCH --mail-user=vincentmillions@gmail.com
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --job-name=model_comparison

# Set environment variables for better memory management
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Array of models (must be redefined in the job script)
models=("wgan" "vae" "standard_gan" "simple_unet")

# Get the model for this array task
model=\${models[\$SLURM_ARRAY_TASK_ID]}

echo "================================================"
echo "SLURM Array Task ID: \$SLURM_ARRAY_TASK_ID"
echo "Model: \$model"
echo "Job ID: \$SLURM_JOB_ID"
echo "Node: \$SLURM_NODELIST"
echo "Time: \$(date)"
echo "================================================"

# Change to the working directory
cd /home/mila/l/letournv/repos/diffusion-model

# Activate conda environment (if needed)
# source /home/mila/l/letournv/miniconda3/etc/profile.d/conda.sh
# conda activate cphoto

# Run the comparison experiment
echo "Starting experiment with model: \$model"

# Build the command
CMD="python train_comparison.py --config-name=comparison_config model=\$model train=\$model logger=True"

# Add debug flag if passed
if [[ "${debug_flag}" != "" ]]; then
    CMD="\$CMD ${debug_flag}"
fi

echo "Executing: \$CMD"
eval \$CMD

echo "Completed experiment with model: \$model"
echo "Time: \$(date)"
EOF

echo "Array job submitted!"
echo "Use 'squeue -u \$USER' to check job status"
echo "Output files will be in slurm/ directory as comparison_<job_id>_<array_index>.out"
echo ""
echo "Usage:"
echo "  ./run_comparison_experiments.sh          # Run all models with full training"
echo "  ./run_comparison_experiments.sh debug    # Run all models in debug mode"
