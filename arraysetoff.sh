#!/bin/bash

# Script to submit array jobs for train3.py with different n_samples values
# Run n jobs for each n_samples value using SLURM array jobs

# Array of n_samples values
n_samples_values=(28 47 261 180 277 192 97 230 52 209 21 273)

# Create slurm directory if it doesn't exist
mkdir -p slurm

# Calculate total number of jobs (2 runs per n_samples value)
n=1
total_jobs=$((${#n_samples_values[@]} * n))
max_array_index=$((total_jobs - 1))

echo "Submitting array job for train3.py..."
echo "Total jobs: ${total_jobs} (array indices 0-${max_array_index})"

sbatch << EOF
#!/bin/bash
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --constraint="32gb|40gb|48gb"
#SBATCH -c 1
#SBATCH --mem=24G
#SBATCH -t 40:00:00
#SBATCH --array=0-${max_array_index}
#SBATCH --output slurm/%A_%a.out
#SBATCH --error slurm/%A_%a.err
#SBATCH --mail-user=vincentmillions@gmail.com
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --job-name=train3_array

# Set environment variables for better memory management
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Array of n_samples values (must be redefined in the job script)
n_samples_values=(28 47 261 180 277 192 97 230 52 209 21 273)

# Calculate which n_samples value and run_id to use based on SLURM_ARRAY_TASK_ID
n_samples_index=\$((SLURM_ARRAY_TASK_ID / 2))
run_id=\$((SLURM_ARRAY_TASK_ID % 2 + 1))
n_samples=\${n_samples_values[\$n_samples_index]}

echo "Array task ID: \$SLURM_ARRAY_TASK_ID"
echo "n_samples_index: \$n_samples_index"
echo "n_samples: \$n_samples"
echo "run_id: \$run_id"

cd /home/mila/l/letournv/repos/diffusion-model
python train3.py training.n_samples=\$n_samples
EOF

echo "Array job submitted!"
echo "Use 'squeue -u \$USER' to check job status"
echo "Output files will be in slurm/ directory as <job_id>_<array_index>.out" 
