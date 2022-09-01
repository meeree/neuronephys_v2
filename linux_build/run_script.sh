#!/bin/bash
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=30g
#SBATCH --cpus-per-task=1
#SBATCH --time=20:00:00
#SBATCH --account=forger0
#SBATCH --partition=gpu
#SBATCH --mail-type=NONE
#SBATCH --gres=gpu:1
#SBATCH --job-name=neuronephys
srun mainS $@
