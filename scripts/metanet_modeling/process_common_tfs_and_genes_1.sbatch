#!/bin/bash

#SBATCH -J process_common_tfs_and_genes_1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=16G
#SBATCH --array=1-36
#SBATCH -o ./logs/process_common_tfs_and_genes_1_%A_%a.out

source activate human_tf_project

tissue=$( sed -n ${SLURM_ARRAY_TASK_ID}p ./tissues.txt )
path=/scratch/mblab/jungw/METANet/data/
binding_threshold_10perc=0.10

python3 -u process_common_tfs_and_genes_1.py \
    --tissue $tissue \
    --path $path \
    --binding_threshold $binding_threshold_10perc
