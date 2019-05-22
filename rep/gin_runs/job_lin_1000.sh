#!/bin/sh

#SBATCH --job-name=lin_pca1000
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err
#SBATCH --mail-type=end
#SBATCH --mail-user=giurgiu@in.tum.de
#SBATCH --nodes=1 --ntasks-per-node=1
#SBATCH --mem-per-cpu=10000
#SBATCH --auks=no
#SBATCH --priority=medium
gt -w tum/rep --run-id 2019-03-15_20-21-47_lasso_model_pca_1000_only_blood --gin-bindings pca_train.n_components=1000 linear_regression_pca.gin /s/project/rep/processed/training_results/linear_regression/
