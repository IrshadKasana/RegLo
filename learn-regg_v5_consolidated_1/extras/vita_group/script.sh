#!/bin/bash 
#SBATCH --job-name=            # create a short name for your job 
#SBATCH -N 1                     # node count 
#SBATCH --ntasks-per-node=1      # total number of tasks per nodes 
#SBATCH --gres=gpu:A100-SXM4:1 
#SBATCH --time=36:00:00          # total run time limit (HH:MM:SS)
#SBATCH --output=console/IMG/MNIST/VQ-EINET.out      
#SBATCH --error=console/IMG/MNIST/VQ-EINET.err       

cd /nlsasfs/home/metalearning/aroof/sahil/einsum-spline-flows 
source ./venv/bin/activate 
cd /nlsasfs/home/metalearning/aroof/sahil/vector-quantized-probabilistic-flow-circuits 
bash scripts/train_mnist.sh 