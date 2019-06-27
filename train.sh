#!/usr/bin/env bash
#SBATCH --time=0-16:15:00
#SBATCH --begin=18:00
#SBATCH -N 1
#SBATCH -C TitanX
#SBATCH --gres=gpu:1

module add cuda90

source activate threes

python stable.py --alg $1 --name $2