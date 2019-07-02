#!/bin/sh

#SBATCH --job-name=corr
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err
#SBATCH --mail-type=end
#SBATCH --mail-user=giurgiu@in.tum.de
#SBATCH --nodes=1 --ntasks-per-node=1
#SBATCH --mem-per-cpu=10000
#SBATCH --cpus-per-task=10
#SBATCH --auks=no
#SBATCH --priority=medium
> /data/ouga/home/ag_gagneur/giurgiu/rep_gagneur/rep/rep/analysis/log.txt
python correlation.py >> /data/ouga/home/ag_gagneur/giurgiu/rep_gagneur/rep/rep/analysis/log.txt
