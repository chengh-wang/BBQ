#!/bin/bash

#SBATCH --output=outputs/%j.out
#SBATCH --error=outputs/%j.err

#SBATCH --nodes=1

#SBATCH --partition=gpu
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --mem=60GB
#SBATCH --time=8:00:00

#SBATCH --mail-type=FAIL
################################
#SBATCH --mail-user=wang.chengh@northeastern.edu

#SBATCH --exclude=c[2160,2162-2175,2176-2183],d1008,d1026

echo "node: $SLURM_NODELIST"


python RL/train_rpp.py --env_name=Ant-v2 --save_dir=./tmp/rpp --rpp_value=False

