#!/bin/bash

#SBATCH -J extract_features_to_files
#SBATCH --mem-per-cpu=2G
#SBATCH --array=1-36
#SBATCH -o ./logs/extract_features_to_files_%A_%a.out

source activate human_tf_project

tissue=$( sed -n ${SLURM_ARRAY_TASK_ID}p ./tissues.txt )

base_path=/scratch/mblab/jungw/METANet/data/
data_path_10perc=${base_path}/${tissue}/METANet/${tissue}_METANet__nlogrank_minmax.txt
output_path_10perc=${base_path}/${tissue}/METANet/features/

mkdir -p $output_path_10perc
python3 extract_features_to_files_3.py \
    --data_path $data_path_10perc \
    --output_path $output_path_10perc
