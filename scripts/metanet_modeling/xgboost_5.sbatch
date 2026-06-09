#!/bin/bash

#SBATCH -J xgboost_metanet_tf_disjoint
#SBATCH --mem-per-cpu=8G
#SBATCH --cpus-per-task=5
#SBATCH --array=1-36
#SBATCH -o ./logs/xgboost_metanet_tf_disjoint_%A_%a.out

source activate human_tf_project


###########################################
### update this to your list of tissues ###
###########################################
tissues_file=./tissues.txt

tissue=$( sed -n ${SLURM_ARRAY_TASK_ID}p $tissues_file )

########################
### UPDATE BASE PATH ###
########################
base_path=/scratch/mblab/jungw/METANet/data/

# log environment for reproducibility
conda env export > $base_path/results/env/environment_${SLURM_ARRAY_JOB_ID}.yml
python3 --version
python3 -c "import xgboost; print('xgboost', xgboost.__version__)"
python3 -c "import sklearn; print('sklearn', sklearn.__version__)"
python3 -c "import hyperopt; print('hyperopt', hyperopt.__version__)"

n_cv_folds=10
seed=42

binding_threshold_10perc=10

###################
### update this ###
###################
binding_threshold=$binding_threshold_10perc

model="METANet"
submodel="metanet"

echo "Running: tissue=$tissue model=$model submodel=$submodel binding_threshold=$binding_threshold seed=$seed n_cv_folds=$n_cv_folds"

python3 -u xgboost.py \
    --tissue $tissue \
    --base_path $base_path \
    --model $model \
    --n_cv_folds $n_cv_folds \
    --seed $seed \
    --submodel $submodel

