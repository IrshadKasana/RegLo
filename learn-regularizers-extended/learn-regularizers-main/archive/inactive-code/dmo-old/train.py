
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from tqdm import tqdm
import seaborn as sns; sns.set(color_codes=True)
sns.set_style("white")
from pdb import set_trace as bp
from meta_module import *
from utils import *
import os
import argparse
import random
import torch.optim as optim
import copy
from torchvision import datasets, transforms
from meta_optimizer import *
from sam import SAM


##################### PARSER ##########################
parser = argparse.ArgumentParser("train.py", description='RNN Optimizer in PyTorch', )
parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                    help='batch size (default: 128)')
parser.add_argument('--seed', type=int, default=1, metavar='N',
                    help='random seed (default: 1)')
parser.add_argument('--optimizee_steps', type=int, default=500, metavar='N',
                    help='number of meta optimizer steps (default: 500)')

parser.add_argument('--truncated_bptt_step', type=int, default=20, metavar='N',
                    help='step at which it truncates bptt (default: 20)')

parser.add_argument('--max_epochs', type=int, default=10000, metavar='N',
                    help='number of epoch (default: 10000)')
parser.add_argument('--hidden_size', type=int, default=20, metavar='N',
                    help='hidden size of the meta optimizer (default: 20)')
parser.add_argument('--num_layers', type=int, default=2, metavar='N',
                    help='number of LSTM layers (default: 2)')
parser.add_argument('--save', type=str, default='/nlsasfs/home/metalearning/nckris/Suraj_LSAIML/L2O-CL/DMO/checkpoints',
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




# args = parser.parse_args()
# print(args)

# if args.cuda_ and torch.cuda.is_available():
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

print(f"Cuda available: {torch.cuda.is_available()} , device name : {device}")

# assert args.optimizee_steps % args.truncated_bptt_step == 0



# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
# os.environ['HYDRA_FULL_ERROR']='1'
USE_CUDA = torch.cuda.is_available()

def w(v):
    if USE_CUDA:
        return v.to(device)
    return v



class MNISTLoss:
    def __init__(self, training=True):
        dataset = datasets.MNIST(
            '/nlsasfs/home/metalearning/nckris/Suraj_LSAIML/datasets/', train=True, download=True,
            transform=torchvision.transforms.ToTensor()
        )
        indices = list(range(len(dataset)))
        np.random.RandomState(10).shuffle(indices)
        if training:
            indices = indices[:len(indices) // 2]
        else:
            indices = indices[len(indices) // 2:]

        self.loader = torch.utils.data.DataLoader(
            dataset, batch_size=128,
            sampler=torch.utils.data.sampler.SubsetRandomSampler(indices))

        self.batches = []
        self.cur_batch = 0
        
    def sample(self):
        if self.cur_batch >= len(self.batches):
            self.batches = []
            self.cur_batch = 0
            for b in self.loader:
                self.batches.append(b)
        batch = self.batches[self.cur_batch]
        self.cur_batch += 1
        return batch

class MNISTNet(MetaModule):
    def __init__(self, layer_size=20, n_layers=1, **kwargs):
        super().__init__()

        inp_size = 28*28
        self.layers = {}
        for i in range(n_layers):
            self.layers[f'mat_{i}'] = MetaLinear(inp_size, layer_size)
            inp_size = layer_size

        self.layers['final_mat'] = MetaLinear(inp_size, 10)
        self.layers = nn.ModuleDict(self.layers)

        self.activation = nn.Sigmoid()
        self.loss = nn.NLLLoss()

    def all_named_parameters(self):
        return [(k, v) for k, v in self.named_parameters()]
    
    def forward(self, loss):
        inp, out = loss.sample()
        inp = w(Variable(inp.view(inp.size()[0], 28*28)))
        out = w(Variable(out))

        cur_layer = 0
        while f'mat_{cur_layer}' in self.layers:
            inp = self.activation(self.layers[f'mat_{cur_layer}'](inp))
            cur_layer += 1

        inp = F.log_softmax(self.layers['final_mat'](inp), dim=1)
        l = self.loss(inp, out)
        return l



#train the learned optimizer with mnist dataset



min_loss= 10000000
for lr in tqdm(sorted([1.0, 0.3, 0.1, 0.03, 0.01, 0.003, 0.001, 0.0003, 0.0001, 0.00003, 0.00001], key=lambda x: np.abs(x - 0.003)), 'all'):
    print('Trying lr:', lr)
    loss, net = fit_optimizer(MNISTLoss, MNISTNet, lr=lr, out_mul=0.1, preproc=True, n_tests=5, n_epochs=10)
    if loss < min_loss:
        min_loss = loss
        best_net = net

    #save all network params here
    checkpoint = {
            'lr': lr,
            'optimizer_state_dict': net,
            'loss': loss.item(),  # Add any relevant information
        }
    torch.save(checkpoint, f'/home/surajk/Mtech_project/l2l/checkpoints_LO/CPpoint_LO_lr_{lr}.pth')




