#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --gres=gpu:titanrtx:2
#SBATCH --mem=250GB
#SBATCH --time=48:00:00
#SBATCH --array=0
#SBATCH --job-name=train
#SBATCH --output=train_%A_%a.out

module purge
module load cuda-10.1

#python -u /misc/vlgscratch4/LakeGroup/emin/temporal-prediction/main.py --cuda --data 'intphys'
python -u /misc/vlgscratch4/LakeGroup/emin/temporal-prediction/main.py --cuda --embedding-model 'rand' --data 'a'

echo "Done"
