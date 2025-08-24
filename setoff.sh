#!/bin/bash
#SBATCH --gres=gpu:40gb
#SBATCH -c 8
#SBATCH --mem=40G
#SBATCH -t 24:00:00                                 
#SBATCH --output slurm/%j.out
#SBATCH --error slurm/%j.err
#SBATCH --mail-user=vincentmillions@gmail.com
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --comment="diffusion"

python $1 
