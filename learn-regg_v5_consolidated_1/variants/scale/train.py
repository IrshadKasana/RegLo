#################################################################

#           Train the optimizer model     

#################################################################
import numpy as np
import tqdm
import argparse
import os
import random
import sys
from pathlib import Path
import torch
import torch.nn.functional as F
import torch.optim as optim
import copy
from torch.autograd import Variable
from torchvision import datasets, transforms
from tqdm import tqdm

from utils import *
from meta_optimizer import *
from model import *
from dataloaders import *
from sam import SAM


import os

######################
#memory leaks
'''
https://pytorch.org/blog/understanding-gpu-memory-1/
'''
import logging
import socket
from datetime import datetime, timedelta

sys.path.append(str(Path(__file__).resolve().parents[2]))

from project_paths import ensure_dir, remap_workspace_path

logging.basicConfig(
   format="%(levelname)s:%(asctime)s %(message)s",
   level=logging.INFO,
   datefmt="%Y-%m-%d %H:%M:%S",
)
logger: logging.Logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)

TIME_FORMAT_STR: str = "%b_%d_%H_%M_%S"


MAX_NUM_OF_MEM_EVENTS_PER_SNAPSHOT: int = 100000

def start_record_memory_history() -> None:
   if not torch.cuda.is_available():
       logger.info("CUDA unavailable. Not recording memory history")
       return

   logger.info("Starting snapshot record_memory_history")
   torch.cuda.memory._record_memory_history()
   
def stop_record_memory_history() -> None:
   if not torch.cuda.is_available():
       logger.info("CUDA unavailable. Not recording memory history")
       return

   logger.info("Stopping snapshot record_memory_history")
   torch.cuda.memory._record_memory_history(enabled=None)

def export_memory_snapshot() -> None:
   if not torch.cuda.is_available():
       logger.info("CUDA unavailable. Not exporting memory snapshot")
       return

   # Prefix for file names.
   host_name = socket.gethostname()
   timestamp = datetime.now().strftime(TIME_FORMAT_STR)
   file_prefix = f"{host_name}_{timestamp}"

   try:
       logger.info(f"Saving snapshot to local file: {file_prefix}.pickle")
       torch.cuda.memory._dump_snapshot(f"{file_prefix}.pickle")
   except Exception as e:
       logger.error(f"Failed to capture memory snapshot {e}")
       return

###################
# from torch.utils.tensorboard import SummaryWriter
# writer = SummaryWriter()



parser = argparse.ArgumentParser("train.py", description='RNN Optimizer in PyTorch', )
parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                    help='batch size (default: 128)')
parser.add_argument('--seed', type=int, default=1, metavar='N',
                    help='random seed (default: 1)')
parser.add_argument('--optimizee_steps', type=int, default=500, metavar='N',
                    help='number of meta optimizer steps (default: 500)')
# parser.add_argument('--lamb', type=float, default=1.0, metavar='N',
#                     help='regularization coefficient')
# parser.add_argument('--eps', type=float, default=1.0, metavar='N',
                    # help='perturbation strength')
parser.add_argument('--truncated_bptt_step', type=int, default=20, metavar='N',
                    help='step at which it truncates bptt (default: 20)')
# parser.add_argument('--updates_per_epoch', type=int, default=2, metavar='N',
#                     help='updates per epoch (default: 2)')
parser.add_argument('--max_epochs', type=int, default=10000, metavar='N',
                    help='number of epoch (default: 10000)')
parser.add_argument('--hidden_size', type=int, default=20, metavar='N',
                    help='hidden size of the meta optimizer (default: 20)')
parser.add_argument('--num_layers', type=int, default=2, metavar='N',
                    help='number of LSTM layers (default: 2)')
parser.add_argument('--save', type=str, default='checkpoints',
                   help="location to optimizer checkpoints")
parser.add_argument('--cuda_', action='store_true', default=True,
                    help='enables CUDA training')


parser.add_argument('--if_cl', action='store_true',
                    help = 'employ CL(True) or not(False)')
parser.add_argument('--dataset', type = str, default= 'mnist',
                    help = 'dataset for the optimizee training')
parser.add_argument('--optimizee_model', type = str, default= 'mlp',
                    help = 'choice of neural network for the optimizee(mlp/deep_mlp/conv/conv_large)')
parser.add_argument('--n_period', type= int, default= 3,
                    help = 'number of times LO is evaluated with the same #inner-iterations')
parser.add_argument('--t_period', type = int, default= 100,
                    help = 'number of epochs LO trained with the same #inner-iteratarions')
parser.add_argument('--eval_epochs', type= int, default = 20,
                    help = 'number of epochs for the evaluation of the optimizers')
parser.add_argument('--if_avreg', action='store_true',
                    help = 'turn on regularization on Optimizer')


args = parser.parse_args()
print(args)

device = torch.device('cuda:0' if args.cuda_ and torch.cuda.is_available() else 'cpu')

print(f"Cuda available: {torch.cuda.is_available()} , device name : {device}")

assert args.optimizee_steps % args.truncated_bptt_step == 0

random.seed(args.seed)
torch.manual_seed(args.seed)

#directory
save_root = ensure_dir(remap_workspace_path(args.save, __file__))


#dataloaders
if (args.dataset == 'mnist'):
    train_data_loader, test_data_loader = data_loader("mnist", batch_size = 32, random_seed = 10)

torch.cuda.empty_cache()

def main():

    loss_min = 99999.9999
    n = 1

    if args.if_cl == True:
        N_train_steps = [100, 200, 500, 1000, 1500, 2000, 2500, 3000]
        N_eval_steps = N_train_steps[1:]
        curriculum_idx = 0
        num_unroll_train = N_train_steps[curriculum_idx]
        num_unroll_eval = N_eval_steps[curriculum_idx]
    
    else:
        num_unroll_train = args.optimizee_steps
        num_unroll_eval = args.optimizee_steps

    # # Start recording memory snapshot history
    # start_record_memory_history()

    meta_model = initialize_model(args.dataset, args.optimizee_model)

    meta_model = meta_model.to(device)


    meta_optimizer = MetaOptimizerSDMO(MetaModel(meta_model), args.num_layers, 1, args.hidden_size)

    meta_optimizer = meta_optimizer.to(device)

    improved = False
    for eph in range(args.max_epochs//args.t_period):
        
        if args.if_avreg:
            reg_train_opt_func(args.truncated_bptt_step,
                               num_unroll_train,
                               args.t_period,
                               train_data_loader,
                               args.dataset,
                               args.optimizee_model,
                               device,
                               meta_optimizer)
        else:    
            train_opt_func(args.truncated_bptt_step,
                        num_unroll_train,
                        args.t_period,
                        train_data_loader,
                        args.dataset,
                        args.optimizee_model,
                        device,
                        meta_optimizer)
        
        torch.cuda.empty_cache()

        # # Create the memory snapshot file
        # export_memory_snapshot()

        # # Stop recording memory snapshot history
        # stop_record_memory_history()

        #evaluation of the optimizer
        n = n + 1

        
        loss_eval = eval_opt_func(num_unroll_eval,
                                train_data_loader,
                                args.dataset,
                                args.optimizee_model,
                                device,
                                meta_optimizer,
                                args.eval_epochs)
        torch.cuda.empty_cache()
        
        if not args.if_cl:
            if loss_eval < loss_min:
                loss_min = loss_eval
                
            #save model here
            if not os.path.exists(save_root):
                os.mkdir(save_root)

            if args.if_avreg == True:
                optimizer_path = os.path.join(save_root, f'SDMO_CL_ckpt_reg.pt')
            else:
                optimizer_path = os.path.join(save_root, f'SDMO_CL_ckpt.pt')

            torch.save(meta_optimizer.state_dict(), optimizer_path)

            continue
        


        #Curriculum Learning

        if loss_eval < loss_min:
            
            loss_min = loss_eval
            improved = True

            #save model here
            if not os.path.exists(save_root):
                os.mkdir(save_root)
            best_meta_optimizer = copy.deepcopy(meta_optimizer)
            
            if args.if_avreg == True:
                print("\n\n\n reg Model Saved \n\n\n")
                optimizer_path = os.path.join(save_root, f'SDMO_CL_ckpt_reg.pt')
            else:
                print("\n\n\n Model Saved \n\n\n")
                optimizer_path = os.path.join(save_root, f'SDMO_CL_ckpt.pt')

            torch.save(meta_optimizer.state_dict(), optimizer_path)

        if args.if_cl == True:
            if n >= args.n_period and improved:

                print(f"\n\nCurriculum stage {curriculum_idx} completed !!\n\n\n")
                n = 1
                curriculum_idx += 1
                if curriculum_idx >= len(N_train_steps):
                    curriculum_idx = -1
                num_unroll_train = N_train_steps[curriculum_idx]
                num_unroll_eval = N_train_steps[curriculum_idx]


                #initial evaluation for next curriculum
                loss_min =  eval_opt_func(num_unroll_eval,
                                        train_data_loader,
                                        args.dataset,
                                        args.optimizee_model,
                                        device,
                                        best_meta_optimizer,
                                        args.eval_epochs)
                torch.cuda.empty_cache()


            elif n >= args.n_period and not improved:
                print(f"No improvement with curriculum {curriculum_idx} : STOPPING Here!")
                break


if __name__ == "__main__":
    main()
