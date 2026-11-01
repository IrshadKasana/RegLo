#!/bin/bash 
#SBATCH --job-name=sdm_lo      # create a short name for your job 
#SBATCH -N 1                     # node count 
#SBATCH --ntasks-per-node=32      # total number of tasks per nodes 
#SBATCH --gres=gpu:A100-SXM4:1 
#SBATCH --time=36:00:00          # total run time limit (HH:MM:SS)
#SBATCH --partition=testp
#SBATCH --output=console/outs/sdm_lo.out
#SBATCH --error=console/errors/sdm_lo.err


cd /nlsasfs/home/metalearning/nckris/Suraj_LSAIML/suraj_env1
source ./bin/activate 
cd /nlsasfs/home/metalearning/nckris/Suraj_LSAIML/chen_2020
python3 train_LO.py
