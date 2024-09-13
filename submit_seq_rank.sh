#!/bin/bash

#SBATCH --job-name=seq_rank
#SBATCH --output=seq_rank.out.%j
#SBATCH --error=seq_rank.out.%j
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --account=engineering
#SBATCH --partition=p5
#SBATCH --priority=normal

python train_sequence_rank.py