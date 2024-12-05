#!/bin/bash

#SBATCH --job-name=sm
#SBATCH --output=sm.out.%j
#SBATCH --error=sm.out.%j
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --account=engineering
#SBATCH --partition=p5
#SBATCH --qos=idle

python train_trl.py