import glob
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

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
os.environ['HYDRA_FULL_ERROR']='1'
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


class MNISTNet2Layer(MNISTNet):
    def __init__(self, *args, **kwargs):
        super().__init__(n_layers=2, *args, **kwargs)


##################################### testing with LO ###############################_______________________________________

#######________training the Optimizer with lr of Adam(MetaOpt) = 0.01_________######

#best loss 56.723411560058594
#best learning rate 0.01

loss, mnist_optimizer = fit_optimizer(MNISTLoss, MNISTNet, lr=0.01, optim_it= 1000, n_tests=20, out_mul=0.1, preproc=True)
print("loss using learned optimizer: ", loss)

torch.save(mnist_optimizer, f'/home/surajk/Mtech_project/l2l/checkpoints_LO/trained_optimizeronMNIST.pth')

mnist_opt_dict = torch.load("/home/surajk/Mtech_project/l2l/checkpoints_LO/trained_optimizeronMNIST.pth")


#-__________________________________________________________#

#fit again to find the grad values and save 
opt = w(Optimizer(preproc=True))
opt.load_state_dict(mnist_opt_dict)
#number of optimization steps for optimizee = 1000
loss_lo, params_lo, gradients = do_fit(opt,None, MNISTLoss, MNISTNet, 1, 1000, out_mul=0.1, should_train=False)

grad_np = np.array(gradients)
file_name = 'gradient_by_LO.npy'
np.save("/home/surajk/Mtech_project/l2l/checkpoints_LO/"+file_name, grad_np)
#-_____________________________________________________________________________________________#








NORMAL_OPTS = [(optim.Adam, {}), (optim.RMSprop, {}), (optim.SGD, {'momentum': 0.9}), (optim.SGD, {'nesterov': True, 'momentum': 0.9})]
OPT_NAMES = ['ADAM', 'RMSprop', 'SGD', 'NAG']


QUAD_LRS = [0.03, 0.01, 1.0, 1.0]
N_TESTS = 100

for i, ((opt, extra_kwargs), lr) in enumerate(zip(NORMAL_OPTS, QUAD_LRS)):
    print(i, opt, lr)


print("for normal opts___________________________________")
fit_data = np.zeros((N_TESTS, 200, len(OPT_NAMES) + 1))
for i, ((opt, extra_kwargs), lr) in enumerate(zip(NORMAL_OPTS, QUAD_LRS)):
    print(f"optimizer using: {opt}, {lr}")
    np.random.seed(0)
    loss_normal, params_normal, gradients_normal = fit_normal(MNISTLoss, MNISTNet, opt, lr=lr, n_tests=N_TESTS, n_epochs=200, **extra_kwargs)
    fit_data[:, :, i] = np.array(loss_normal)
    grad_np = np.array(gradients_normal)
    file_name = f'gradient_by_{opt}_{lr}.npy'
    np.save("/home/surajk/Mtech_project/l2l/checkpoints_LO/"+file_name, grad_np)
    
    if i == 2:
        torch.save(params_normal, f'/home/surajk/Mtech_project/l2l/checkpoints_LO/sgd_model_params.pth')

print(fit_data)






#use the do_fit function to fit a test task for 'n_tests' many number of times
@cache.cache
def get_fit_dict_test(n_tests, opt_dict, *args, **kwargs):
    opt = w(Optimizer(preproc=True))
    opt.load_state_dict(opt_dict)
    np.random.seed(0)
    
    return [do_fit(opt, *args, **kwargs)[0] for _ in tqdm(range(n_tests), 'optimizer')]

print("\n \n fitting using LO again, for losses:_____________________________")
fit_data[:, :, len(OPT_NAMES)] = np.array(get_fit_dict_test(N_TESTS, mnist_opt_dict, None, MNISTLoss, MNISTNet, 1, 200, out_mul=0.1, should_train=False))





# ####################################
# #fit again to find the grad values and save 
# opt = w(Optimizer(preproc=True))
# opt.load_state_dict(mnist_opt_dict)
# #number of optimization steps for optimizee = 1000
# loss_lo, params_lo, gradients = do_fit(opt,None, MNISTLoss, MNISTNet, 1, 1000, out_mul=0.1, should_train=False)

# grad_file = open("grad_values_paramsOptimizee_LO.txt", "w")
# for l in gradients:
#     grad_file.write(l)
#     grad_file.write("\n")
# grad_file.close()
# ####################################


###############save all the losses in every type of optimizer#####################

file_name = 'loss_out.npy'
np.save("/home/surajk/Mtech_project/l2l/checkpoints_LO/"+file_name, fit_data)



# ax = plt.figure()
# def mean_std_plot(ax, data,**kw):
#     color=['r', 'b', 'g', 'k', 'y']
#     for i in range (fit_data.shape[2]):
#         colr = color[i]
#         data = fit_data[:,:,i]
#         x = np.arange(data.shape[1])
#         est = np.mean(data, axis=0)
#         sd = np.std(data, axis=0)
#         cis = (est - sd, est + sd)
#         ax.fill_between(x,cis[0],cis[1],alpha=0.2, **kw, color = colr)
#         ax.plot(x,est,**kw)
#         ax.margins(x=0)

# mean_std_plot(ax, data=fit_data[:, :, :])



# ax = sns.tsplot(data=fit_data[:, :, :], condition=OPT_NAMES + ['LSTM'], linestyle='--', color=['r', 'b', 'g', 'k', 'y'])
# ax.lines[-1].set_linestyle('-')
# ax.legend()
# plt.yscale('log')
# plt.xlabel('steps')
# plt.ylabel('loss')
# plt.title('MNIST')
# plt.savefig("mnist_LO_vs_others1.png")




#########################################################################
#plot loss connecting sgd solution and lo_solution

# loss = optimizee(target)

##########################################################################
#train on different architecture

