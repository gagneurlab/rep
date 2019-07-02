#!/bin/sh

#SBATCH --job-name=lin_simple
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err
#SBATCH --mail-type=end
#SBATCH --mail-user=giurgiu@in.tum.de
#SBATCH --nodes=1 --ntasks-per-node=1
#SBATCH --mem-per-cpu=10000
#SBATCH --cpus-per-task=10
#SBATCH --auks=no
#SBATCH --priority=medium
gt -w tum/rep linear_regression_simple.gin /s/project/rep/processed/training_results/linear_regression/
