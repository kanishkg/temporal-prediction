#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --gres=gpu:1080ti:1
#SBATCH --mem=100GB
#SBATCH --time=48:00:00
#SBATCH --array=0
#SBATCH --job-name=eval_adept
#SBATCH --output=eval_adept_%A_%a.out

module purge
module load cuda-10.1

python -u /misc/vlgscratch4/LakeGroup/emin/temporal-prediction/evaluate_adept.py \
--embedding-model 'say' \
--dynamics-data 'a' \
--data-dir '/misc/vlgscratch4/LakeGroup/emin/ADEPT/test'

echo "Done"
