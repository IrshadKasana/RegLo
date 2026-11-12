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
from torchvision.datasets import ImageFolder

import torchvision.transforms as tt
import ssl
ssl._create_default_https_context = ssl._create_unverified_context


# dataset_url = 'https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz'
# # od.download(dataset_url)
# data_dir = '/home/surajk/Mtech_project/l2l/cifar-100-python'


# train_tfms = tt.Compose([tt.RandomCrop(32, padding=4,padding_mode='reflect'), 
#                          tt.RandomHorizontalFlip(), 
#                          tt.ToTensor()])

# dataset = ImageFolder(data_dir+'/train', train_tfms)

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ['HYDRA_FULL_ERROR']='1'
USE_CUDA = torch.cuda.is_available()

def w(v):
    if USE_CUDA:
        return v.cuda()
    return v



class CIFAR100Loss:
    def __init__(self, training=True):
        self.training = training

        dataset = datasets.CIFAR100(
            './cifar100', train=True, download=True,
            transform=torchvision.transforms.ToTensor()
        )
         #/home/surajk/Mtech_project/l2l/cifar100---> path where the dataset is downloaded

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



list_files = os.listdir('/home/surajk/Mtech_project/l2l/checkpoints_LO')
max = 1000000000000
for file_n in list_files:
    dict_params = torch.load("/home/surajk/Mtech_project/l2l/checkpoints_LO/"+file_n)
    print(dict_params['lr'], dict_params['loss'])

    print("\n \n ###############################################################\n \n")
    if (dict_params['loss'] < max):
        max = dict_params['loss']
        best_opt = dict_params['optimizer_state_dict']
        best_lr = dict_params['lr']

print("best loss", max)
print("best learning rate", best_lr)

print("\n \n ###############################################################\n \n")





@cache.cache
def get_fit_dict_test(opt_dict, *args, **kwargs):
    opt = w(Optimizer(preproc=True))
    opt.load_state_dict(opt_dict)
    np.random.seed(0)
    return do_fit(opt, *args, **kwargs)
loss_and_params = get_fit_dict_test(best_opt, None, CIFAR100Loss, MNISTNet, 1, 200, 200, out_mul=0.1, should_train=False)

print(loss_and_params)

