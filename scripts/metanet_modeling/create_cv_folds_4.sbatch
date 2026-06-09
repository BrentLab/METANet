#!/bin/bash

#SBATCH -J create_cv_folds
#SBATCH --mem-per-cpu=8G
#SBATCH --cpus-per-task=1
#SBATCH --array=1-36#1-36
#SBATCH -o ./logs/create_cv_folds_%A_%a.out

source activate human_tf_project

tissue=$( sed -n ${SLURM_ARRAY_TASK_ID}p ./tissues.txt )
base_path=/scratch/mblab/jungw/METANet/data/
binding_threshold_20perc=20
binding_threshold_10perc=10
binding_threshold_05perc=5
binding_threshold=$binding_threshold_10perc
model="METANet"

python3 -u create_cv_folds.py \
    --tissue $tissue \
    --base_path $base_path \
    --model $model \
    --num_CV_folds 10 \
    --binding_threshold $binding_threshold
