#!/bin/sh

config_gin="linear_regression.gin"
outdir="/s/project/rep/processed/training_results/linear_regression/"
declare -a pca_comp=(10 20 50 100 200 1000)

for p  in "${pca_comp[@]}"
do
	# set param
	sed -i -e 's/n_components = 10/n_components = '$p'/g' $config_gin
	file="job_lin_${p}.sh"
	name="lin_pca${p}"
	project="tum/rep"
	description="lasso_model,pca_${p},only_blood,20iter"
	
	# reset content
	touch $file
	> $file
	
	echo "#!/bin/sh" >> $file
	echo "" >> $file
	echo "#SBATCH --job-name=${name}" >> $file
	#echo "#SBATCH --begin=now+60minute" >> $file
	echo "#SBATCH --output=slurm_%j.out" >> $file
	echo "#SBATCH --error=slurm_%j.err" >> $file
	echo "#SBATCH --mail-type=end" >> $file
	echo "#SBATCH --mail-user=giurgiu@in.tum.de" >> $file
	echo "#SBATCH --nodes=1 --ntasks-per-node=1" >> $file
	echo "#SBATCH --mem-per-cpu=10000" >> $file
	echo "#SBATCH --auks=no" >> $file
	echo "#SBATCH --priority=medium" >> $file
	echo "gt -w $project --run-id ${description} $config_gin $outdir" >> $file
	
	sbatch $file
	# reset param
	sed -i -e 's/n_components = '${p}'/n_components = 10/g' $config_gin
done
