#!/bin/bash

#SBATCH -J feature_transformation_2
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=16G
#SBATCH --array=1-36 #36
#SBATCH -o ./logs/feature_transformation_2_%A_%a.out


export R_LIBS_SITE=/scratch/mblab/jungw/renv/4.4/
export R_LIBS=/scratch/mblab/jungw/renv/4.4/
export TMP=/scratch/mblab/jungw/R_TMP/
export TMPDIR=/scratch/mblab/jungw/R_TMP/
eval $(spack load --sh r@4.4.0)

tissue=$( sed -n ${SLURM_ARRAY_TASK_ID}p ./tissues.txt )
base_path=/scratch/mblab/jungw/METANet/data/
binding_threshold_10perc=10

Rscript feature_transformation_2.R \
    $tissue \
    $base_path \
    $binding_threshold_10perc
