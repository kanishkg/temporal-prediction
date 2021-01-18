#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --gres=gpu:2080ti:2
#SBATCH --mem=180GB
#SBATCH --time=48:00:00
#SBATCH --array=0
#SBATCH --job-name=eval_intphys
#SBATCH --output=eval_intphys_%A_%a.out

module purge
module load cuda-10.1

python -u /misc/vlgscratch4/LakeGroup/emin/temporal-prediction/evaluate_intphys.py \
--embedding-model 'in' \
--dynamics-data 'a'

echo "Done"
