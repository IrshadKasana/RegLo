
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

from grad_operations import *

from torchvision import datasets

import os


os.environ['CUDA_VISIBLE_DEVICES'] = '1'
USE_CUDA = torch.cuda.is_available()

def w(v):
    if USE_CUDA:
        return v.cuda()
    return v



class MNISTLoss:
    def __init__(self, training=True):
        dataset = datasets.MNIST(
            '/home/surajk/Mtech_project/l2l/', train=True, download=True,
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








#############################################################################################################

#We will use the best_net on CIFAR 100

# class CIFAR100Loss:
#     def __init__(self, training=True):
#         self.training = training

#         dataset = datasets.CIFAR100(
#             '/home/surajk/Mtech_project/l2l/cifar100', train=self.training, download=True, 
#             transform=torchvision.transforms.ToTensor()
#         )
#          #/home/surajk/Mtech_project/l2l/cifar100---> path where the dataset is downloaded

#         indices = list(range(len(dataset)))
#         np.random.RandomState(10).shuffle(indices)
#         if training:
#             indices = indices[:len(indices) // 2]
#         else:
#             indices = indices[len(indices) // 2:]

#         self.loader = torch.utils.data.DataLoader(
#             dataset, batch_size=128,
#             sampler=torch.utils.data.sampler.SubsetRandomSampler(indices))

#         self.batches = []
#         self.cur_batch = 0
        
#     def sample(self):
#         if self.cur_batch >= len(self.batches):
#             self.batches = []
#             self.cur_batch = 0
#             for b in self.loader:
#                 self.batches.append(b)
#         batch = self.batches[self.cur_batch]
#         self.cur_batch += 1
#         return batch





# @cache.cache
# def get_fit_dict_test(opt_dict, *args, **kwargs):
#     opt = w(Optimizer(preproc=True))
#     opt.load_state_dict(opt_dict)
#     np.random.seed(0)
#     return do_fit(opt, *args, **kwargs)
# loss_and_params = get_fit_dict_test(best_net, None, CIFAR100Loss, MNISTNet, 1, 200, 200, out_mul=0.1, should_train=False)

# lowest_loss, best_params = loss_and_params[-1]



# print(best_params)



# #solution using SGD and others

# NORMAL_OPTS = [(optim.Adam, {}), (optim.RMSprop, {}), (optim.SGD, {'momentum': 0.9}), (optim.SGD, {'nesterov': True, 'momentum': 0.9})]
# OPT_NAMES = ['ADAM', 'RMSprop', 'SGD', 'NAG']

# for opt, kwargs in NORMAL_OPTS:
#     print(f"\n \n \n Trying {opt}, finding best learning rate and the corresponding network")
#     best_loss, best_net, best_lr = find_best_lr_normal(MNISTLoss, MNISTNet(training=False), opt, **kwargs, n_tests=10)
    
