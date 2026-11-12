#!/bin/bash 
#SBATCH --job-name=smWdAsamDMO      # create a short name for your job 
#SBATCH -N 1                     # node count 
#SBATCH --ntasks-per-node=1      # total number of tasks per nodes 
#SBATCH --gres=gpu:A100-SXM4:1
#SBATCH --time=72:00:00          # total run time limit (HH:MM:SS)
#SBATCH --partition=testp
#SBATCH --output=console/outs/0203train_dmo_cl_adsam_smoothL05E10WD005_reg.out
#SBATCH --error=console/errors/0203train_dmo_cl_adsam_smoothL05E10WD005_reg.err

cd /nlsasfs/home/metalearning/nckris/Suraj_LSAIML/suraj_env1
source ./bin/activate 
cd /nlsasfs/home/metalearning/nckris/Suraj_LSAIML/L2O-CL/DMO
python3 train.py --if_cl --if_smooth --lamb=0.5 --eps=1.0 --if_avreg --weight_decay=0.05 --adaptive --model_name=0203dmo_cl_adsamWD005_smoothL05E10_reg