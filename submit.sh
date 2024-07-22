#!/bin/bash

#SBATCH --job-name=sm
#SBATCH --output=sm.out.%j
#SBATCH --error=sm.out.%j
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --account=engineering
#SBATCH --partition=p5
#SBATCH --priority=normal

python train.py --wandb --run_name adafactor_fix