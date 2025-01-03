#!/bin/bash

#SBATCH --job-name=sm
#SBATCH --output=sm.out.%j
#SBATCH --error=sm.out.%j
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --account=engineering
#SBATCH --partition=p5
#SBATCH --qos=idle

export WANDB_BASE_URL=https://api.wandb.ai

echo "Visible CUDA devices: $CUDA_VISIBLE_DEVICES"
echo "WANDB base URL: $WANDB_BASE_URL"

python train_trl.py --config codecontests_cot --wandb