#!/bin/bash

#SBATCH --job-name=synth
#SBATCH --output=synth.out.%j
#SBATCH --error=synth.out.%j
#SBATCH --nodes=1
#SBATCH --time=4:00:00
#SBATCH --account=engineering
#SBATCH --partition=cpu24

python generate_synthetic_data.py