#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --job-name=stratification
#SBATCH --mem=10G
#SBATCH --partition=short
#SBATCH --chdir=/home/username/stratification/
#SBATCH --output=/home/username/logs/%x.%j.out

source ~/.bashrc
module load anaconda3/3.7
source activate lib_name
python main.py $n_iter_selec $k_feat "$pre_filter" "$seed" $clf_only $n_iter_classif
