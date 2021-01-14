#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --gres=gpu:v100:2
#SBATCH --mem=250GB
#SBATCH --time=48:00:00
#SBATCH --array=0
#SBATCH --job-name=train
#SBATCH --output=train_%A_%a.out

module purge
module load cuda-10.1

#python -u /misc/vlgscratch4/LakeGroup/emin/temporal-prediction/main.py --cuda --embedding-model 'say' --data 's' --batch-size 100 --dropout 0.05
#python -u /misc/vlgscratch4/LakeGroup/emin/temporal-prediction/main.py --cuda --embedding-model 'say' --data 'a' --batch-size 256 --dropout 0.05
#python -u /misc/vlgscratch4/LakeGroup/emin/temporal-prediction/main.py --cuda --embedding-model 'say' --data 'y' --batch-size 256 --dropout 0.05

python -u /misc/vlgscratch4/LakeGroup/emin/temporal-prediction/main.py --cuda --embedding-model 'in' --data 's' --batch-size 100 --dropout 0.1
python -u /misc/vlgscratch4/LakeGroup/emin/temporal-prediction/main.py --cuda --embedding-model 'in' --data 'a' --batch-size 256 --dropout 0.1
python -u /misc/vlgscratch4/LakeGroup/emin/temporal-prediction/main.py --cuda --embedding-model 'in' --data 'y' --batch-size 256 --dropout 0.1

echo "Done"
