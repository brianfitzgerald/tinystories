#!/bin/bash

#SBATCH --job-name=smo
#SBATCH --output=sm.out.%j
#SBATCH --error=sm.out.%j
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --account=engineering
#SBATCH --partition=h80i
#SBATCH --priority=normal


python train.py --wandb