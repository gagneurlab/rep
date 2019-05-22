#!/bin/sh

#SBATCH --job-name=lin_pca200
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err
#SBATCH --mail-type=end
#SBATCH --mail-user=giurgiu@in.tum.de
#SBATCH --nodes=1 --ntasks-per-node=1
#SBATCH --mem-per-cpu=10000
#SBATCH --auks=no
#SBATCH --priority=medium
gt -w tum/rep --run-id 2019-03-15_17-34-16_lasso_model_pca_200_only_blood --gin-bindings pca_train.n_components=200 linear_regression_pca.gin /s/project/rep/processed/training_results/linear_regression/
#!/bin/sh

#SBATCH --job-name=lin_pca200
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err
#SBATCH --mail-type=end
#SBATCH --mail-user=giurgiu@in.tum.de
#SBATCH --nodes=1 --ntasks-per-node=1
#SBATCH --mem-per-cpu=10000
#SBATCH --auks=no
#SBATCH --priority=medium
gt -w tum/rep --run-id 2019-03-15_17-37-11_lasso_model_pca_200_only_blood --gin-bindings pca_train.n_components=200 linear_regression_pca.gin /s/project/rep/processed/training_results/linear_regression/
#!/bin/sh

#SBATCH --job-name=lin_pca200
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err
#SBATCH --mail-type=end
#SBATCH --mail-user=giurgiu@in.tum.de
#SBATCH --nodes=1 --ntasks-per-node=1
#SBATCH --mem-per-cpu=10000
#SBATCH --auks=no
#SBATCH --priority=medium
gt -w tum/rep --run-id 2019-03-19_19-30-36_lasso_model_pca_200_only_blood --gin-bindings pca_train.n_components=200 linear_regression_pca.gin /s/project/rep/processed/training_results/linear_regression/
#!/bin/sh

#SBATCH --job-name=lin_pca200
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err
#SBATCH --mail-type=end
#SBATCH --mail-user=giurgiu@in.tum.de
#SBATCH --nodes=1 --ntasks-per-node=1
#SBATCH --mem-per-cpu=10000
#SBATCH --auks=no
#SBATCH --priority=medium
gt -w tum/rep --run-id 2019-03-19_19-38-07_lasso_model_pca_200_only_blood --gin-bindings pca_train.n_components=200 linear_regression_pca.gin /s/project/rep/processed/training_results/linear_regression/
