#!/bin/bash
#SBATCH --job-name=unet-train
#SBATCH --partition=comp3710
#SBATCH --gres=gpu:1
#SBATCH --mem=0
#SBATCH --time=03:00:00
#SBATCH --output=train_%j.out
#SBATCH --error=train_%j.err

# Activate PyTorch environment
source $HOME/miniconda3/bin/activate
conda activate torch

# Go to project directory
cd $HOME/PatternAnalysis-2026/recognition/Improved_UNet_HipMRI

# Show GPU information
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
nvidia-smi

# Start training
python -u train.py

