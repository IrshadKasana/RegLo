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
import json
from utils import *
from meta_optimizer import *
from model import *
from dataloaders import *


import os

from datetime import datetime

sys.path.append(str(Path(__file__).resolve().parents[2]))

from project_paths import ensure_dir, remap_workspace_path

parser = argparse.ArgumentParser("optimizee_train.py", description='Train optimizee with RNN Optimizer in PyTorch', )
parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                    help='batch size (default: 128)')
parser.add_argument('--seed', type=int, default=10, metavar='N',
                    help='random seed (default: 10)')
parser.add_argument('--optimizee_epochs', type=int, default=1000, metavar='N',
                    help='number of training steps (default: 500)')

parser.add_argument('--hidden_size', type=int, default=20, metavar='N',
                    help='hidden size of the meta optimizer (default: 20)')
parser.add_argument('--num_layers', type=int, default=2, metavar='N',
                    help='number of LSTM layers (default: 2)')
parser.add_argument('--save', type=str, default='results',
                   help="location to results folder")
parser.add_argument('--cuda_', action='store_true', default=True,
                    help='enables CUDA training')

parser.add_argument('--dataset', type = str, default= 'mnist',
                    help = 'dataset for the optimizee training')
parser.add_argument('--optimizee_model', type = str, default= 'mlp',
                    help = 'choice of neural network for the optimizee(mlp/mlp_Relu/deep_mlp/conv/conv_large)')
parser.add_argument('--n_training', type = int, default = 5,
                    help = 'number of times the same optimizee training is done')
parser.add_argument('--if_avreg', action='store_true',
                    help = 'turn on regularization on Optimizer')


args = parser.parse_args()
print(args)

device = torch.device('cuda:0' if args.cuda_ and torch.cuda.is_available() else 'cpu')

print(f"Cuda available: {torch.cuda.is_available()} , device name : {device}")

random.seed(args.seed)
torch.manual_seed(args.seed)




#directory
save_root = ensure_dir(remap_workspace_path(args.save, __file__))




#defining optimizee
dataset = args.dataset
model_name = args.optimizee_model

meta_model = initialize_model(dataset, model_name)
meta_model = meta_model.to(device)



#optimizer checkpoint read

path = remap_workspace_path("checkpoints", __file__)
if args.if_avreg:
    path = path / "SDMO_CL_ckpt_reg.pt"
    
else:
    path = path / "SDMO_CL_ckpt.pt"


#load the optimizer
num_layers = args.num_layers
hidden_size = args.hidden_size
meta_optimizer = MetaOptimizerSDMO(MetaModel(meta_model), num_layers, 1, hidden_size)
meta_optimizer.load_state_dict(torch.load(path))
meta_optimizer.reset_lstm(keep_states=False, model=meta_model)
meta_optimizer = meta_optimizer.to(device)



def accuracy(outputs, labels):

    _, predicted = torch.max(outputs, 1)
    # print(outputs.size(), predicted.size())
    correct = (predicted == labels).sum().item()
    total = labels.size(0)
    acc = correct / total
    return acc


def train_optimizee(epochs, train_data_loader, test_data_loader,model,meta_optimizer):
    loss_arr = []
    loss_val_arr = []
    acc_val_arr = []
    train_iter = iter(train_data_loader)
    test_iter = iter(test_data_loader)
    for j in tqdm(range(epochs), "epoch: "):
        try:
            x, y = next(train_iter)
        except StopIteration:
            train_iter = iter(train_data_loader)
            x, y = next(train_iter)
        x, y = x.to(device), y.to(device)

        # First we need to compute the gradients wrt model parameters at a point
        f_x = model(x)
        loss = F.nll_loss(f_x, y)

        model.zero_grad()
        loss.backward()

        loss_arr.append(loss.item())

        #Update the values of the parameters using the meta optimizer, meta_model is the new model params
        with torch.no_grad():
            meta_optimizer.meta_update(model)

        #####validating......
        total_correct = 0
        total_count = 0
        v_loss = 0
        with torch.no_grad():
            for x, y in test_data_loader:
                x, y = x.to(device), y.to(device)
                f_x = model(x)
                v_loss += F.nll_loss(f_x, y).item()
                _, predicted = torch.max(f_x, 1)
                total_correct += (predicted == y).sum().item()
                total_count += y.size(0)

        v_acc = total_correct / total_count if total_count else 0.0
        loss_val_arr.append(v_loss/len(test_data_loader))
        acc_val_arr.append(v_acc)


    return loss_arr,loss_val_arr,acc_val_arr



########################### Training ###############################
#epochs = args.n_training

train_data_loader, test_data_loader = data_loader_optimizee(dataset, batch_size = args.batch_size, random_seed = args.seed)
train_iter = iter(train_data_loader)
instances_loss_tr = []
instances_loss_ts = []
instances_acc_ts = []

for i in range (args.n_training):
    print(f"instance {i} ---------------------------------- \n")
    model = initialize_model(dataset, model_name)
    model = model.to(device)
    
    loss_train_arr, loss_test_arr, acc_test_arr  = train_optimizee(args.optimizee_epochs, test_data_loader, test_data_loader,model,meta_optimizer)
    instances_loss_tr.append(loss_train_arr)
    instances_loss_ts.append(loss_test_arr)
    instances_acc_ts.append(acc_test_arr)


json_dict = {"instances_loss_tr" : instances_loss_tr, "instances_loss_ts" : instances_loss_ts, "instances_acc_ts": instances_acc_ts}

current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

if args.if_avreg:
    reg = '_reg_'
else:
    reg = '_no_reg_'
    
output_dir = ensure_dir(save_root / "json")
file_name = output_dir / f"SDMO{model_name}{dataset}{reg}{current_datetime}.json"

with open(file_name, "w") as json_file:
    json.dump(json_dict, json_file)
