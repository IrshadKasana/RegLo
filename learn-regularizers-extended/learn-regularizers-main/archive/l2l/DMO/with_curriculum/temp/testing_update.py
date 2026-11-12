
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
        inp, out = loss.sample() #uses the MNISTLoss to output a batch (X, y)
        inp = w(Variable(inp.view(inp.size()[0], 28*28)))
        out = w(Variable(out))

        cur_layer = 0
        while f'mat_{cur_layer}' in self.layers:
            inp = self.activation(self.layers[f'mat_{cur_layer}'](inp))
            cur_layer += 1

        inp = F.log_softmax(self.layers['final_mat'](inp), dim=1)
        l = self.loss(inp, out)
        return l

def do_fit_new(opt_net, meta_opt, target_cls, target_to_opt, unroll, optim_it, out_mul, should_train=True):

    #use the should_train parameter to evaluate the task and not train the

    if should_train:
        opt_net.train()
    else:
        opt_net.eval()   #switch for some specific layers/parts of the model that behave differently during training and inference (evaluating) time. For example, Dropouts Layers, BatchNorm Layers etc.
        unroll = 1

    target = target_cls(training=should_train)  #it is the dataset, optimizee(target) yields loss
    optimizee = w(target_to_opt())
    n_params = 0
    #velocity
    velocity = {}
    for name, p in optimizee.all_named_parameters():
        n_params += int(np.prod(p.size()))
        cur_sz = int(np.prod(p.size()))
        #initialize velocity
        velocity[name] = torch.zeros_like(p.data.view(cur_sz, 1))
    hidden_states = [w(Variable(torch.zeros(n_params, opt_net.hidden_sz))) for _ in range(2)]
    cell_states = [w(Variable(torch.zeros(n_params, opt_net.hidden_sz))) for _ in range(2)]
    all_losses_ever = []
    if should_train:
        meta_opt.zero_grad()

    #all_losses is the loss for LO(sum of all the losses till unroll)
    all_losses = None



    ###############  keep track of the gradient and update values at the inference time  ########################

    grad_list_test = []
    update_list_test = []
    update_list_test_sgd = []
    

    ##############################

    for iteration in tqdm(range(1, optim_it + 1), "inner-training step"):  #optim_it = number of iterations the optimizee is trained using LO(inner training steps)
        #the loss is always found using all the datapoints (not single or batch loss)
        loss = optimizee(target)

        if all_losses is None:
            all_losses = loss
        else:
            all_losses += loss



        all_losses_ever.append(loss.data.cpu().numpy())

        #gradients wrt all the parameters are updated
        loss.backward(retain_graph=should_train)

        offset = 0
        result_params = {}
        hidden_states2 = [w(Variable(torch.zeros(n_params, opt_net.hidden_sz))) for _ in range(2)]
        cell_states2 = [w(Variable(torch.zeros(n_params, opt_net.hidden_sz))) for _ in range(2)]



        grad_list_each_step = None
        update_list_each_step = None
        update_list_each_step_sgd = None


        #updates for each parameter of optimizee is output by using the LO
        for name, p in optimizee.all_named_parameters():
            cur_sz = int(np.prod(p.size()))
            # We do this so the gradients are disconnected from the graph but we still get
            # gradients from the rest
            gradients = detach_var(p.grad.view(cur_sz, 1))
            param_val = p.data.view(cur_sz, 1)



            #update velocity for momentum sgd
            
            velocity[name] = 0.9 * velocity[name] + gradients
            
              

            # print("gradient_named_param_shape: ", gradients.size())


            ###############  keep track of the gradient values at the inference time  ########################

            if should_train==False:
                if grad_list_each_step==None:
                    grad_list_each_step = gradients
                else:
                    grad_list_each_step = torch.cat((grad_list_each_step, gradients),0)


            ############################################
            # UPDATES ####

            updates, new_hidden, new_cell = opt_net(
                gradients,
                [h[offset:offset+cur_sz] for h in hidden_states],
                [c[offset:offset+cur_sz] for c in cell_states]
            )

            ###############  keep track of the updates the inference time  ########################

            if should_train==False:
                #considering momentum in sgd

                if update_list_each_step==None:
                    update_list_each_step = updates
                    update_list_each_step_sgd = param_val - 1.0*gradients
                            

                else:
                    update_list_each_step = torch.cat((update_list_each_step, updates),0)
                    update_list_each_step_sgd = torch.cat((update_list_each_step_sgd, param_val - 1.0*gradients),0)


            for i in range(len(new_hidden)):
                hidden_states2[i][offset:offset+cur_sz] = new_hidden[i]
                cell_states2[i][offset:offset+cur_sz] = new_cell[i]
            result_params[name] = p + updates.view(*p.size()) * out_mul
            result_params[name].retain_grad()

            offset += cur_sz


#######   unroll when necessary to update the params of the LO   ############
        if iteration % unroll == 0:

            #this process trains the LO during each unroll
            if should_train:
                meta_opt.zero_grad()
                all_losses.backward()  #updates the grad values for optimizees parameters at unrolling (during test time: unroll =1)
                meta_opt.step()   #is the optimizer used to train the LO say Adam

            #make the loss for LO, zero since you need to find this sum again for next unroll step
            all_losses = None


            #need to update the grads for optimizee parameters too since they also are connected to the RNN in the computational graph
            #reducing them to zero is necessary
            optimizee = w(target_to_opt())
            optimizee.load_state_dict(result_params)
            optimizee.zero_grad()
            hidden_states = [detach_var(v) for v in hidden_states2]
            cell_states = [detach_var(v) for v in cell_states2]


#######  steps when no unrolling happens ##########################
        else:
            for name, p in optimizee.all_named_parameters():
                rsetattr(optimizee, name, result_params[name])
            assert len(list(optimizee.all_named_parameters()))
            hidden_states = hidden_states2
            cell_states = cell_states2

        # print("grad_shape_each_epoch:", grad_list_each_step.size())
        grad_list_test.append(grad_list_each_step)
        update_list_test.append(update_list_each_step)
        update_list_test_sgd.append(update_list_each_step_sgd)

    #all_losses_ever is the loss that optimizee gives after each iteration
    return [all_losses_ever, optimizee, grad_list_test, update_list_test, update_list_test_sgd]

mnist_opt_dict = torch.load("/home/surajk/Mtech_project/l2l/checkpoints_LO/trained_optimizeronMNIST.pth")
opt = w(Optimizer(preproc=True))
opt.load_state_dict(mnist_opt_dict)

loss_lo, params_lo, gradients_lo, updates_lo, update_lo_sgd = do_fit_new(opt,None, MNISTLoss, MNISTNet, 1, 500, out_mul=0.1, should_train=False)

opt = w(Optimizer(preproc=True))
opt.load_state_dict(mnist_opt_dict)

LO_update_sgd_cos_s = []

for i in tqdm(range(len(updates_lo)), "iterations"):
    u1 = torch.reshape(updates_lo[i], (-1,1)).cpu().detach().numpy()
    u2 = torch.reshape(update_lo_sgd[i], (-1,1)).cpu().detach().numpy()
    val = np.dot(np.transpose(u1), u2)/(np.linalg.norm(u1)*np.linalg.norm(u2))
    LO_update_sgd_cos_s.append(val[0][0])


file_ = open("/home/surajk/Mtech_project/l2l/checkpoints_LO/update_cos_c.txt", "w+")
for val in LO_update_sgd_cos_s:
    file_.write(str(val))
    file_.write('\n')

file_.close()