#!/bin/sh

config_gin="linear_regression_simple.gin"
outdir="/s/project/rep/processed/training_results/linear_regression/"

p="simple"

# description
file="job_lin_${p}.sh"
name="lin_${p}"
project="tum/rep"
#description="$(date +%F_%H-%M-%S)_lin_reg_${p}_only_blood"
description='lin_reg_simple_only_blood'

# slurm config
echo "#!/bin/sh" > $file
echo "" >> $file
echo "#SBATCH --job-name=${name}" >> $file
echo "#SBATCH --output=slurm_%j.out" >> $file
echo "#SBATCH --error=slurm_%j.err" >> $file
echo "#SBATCH --mail-type=end" >> $file
echo "#SBATCH --mail-user=giurgiu@in.tum.de" >> $file
echo "#SBATCH --nodes=1 --ntasks-per-node=1" >> $file
echo "#SBATCH --mem-per-cpu=10000" >> $file
echo "#SBATCH --cpus-per-task=10" >> $file
echo "#SBATCH --auks=no" >> $file
echo "#SBATCH --priority=medium" >> $file
echo "gt -w $project $config_gin $outdir" >> $file

rm -rf "${outdir}/${description}"
chmod 755 $file
sbatch $file
