import glob
import numpy as np
import torch
import torch.nn as nn

import torch.optim as optim
from torch.autograd import Variable

from tqdm import tqdm


import copy
import joblib
from torchvision import datasets


from pdb import set_trace as bp
from l_optimizer import Optimizer
from meta_module import *


import functools



USE_CUDA = torch.cuda.is_available()

def w(v):
    if USE_CUDA:
        return v.cuda()
    return v

cache = joblib.Memory(location='_cache', verbose=0)

def detach_var(v):
    var = w(Variable(v.data, requires_grad=True))
    var.retain_grad()
    return var

###############################################################################################


#add attribute values to the object class if not assigned in the form of a 
# function setattributes(object_name, attribute_name, value_of_the_attribute)

def rsetattr(obj, attr, val):
    pre, _, post = attr.rpartition('.')
    return setattr(rgetattr(obj, pre) if pre else obj, post, val)

# using wonder's beautiful simplification: https://stackoverflow.com/questions/31174295/getattr-and-setattr-on-nested-objects/31174427?noredirect=1#comment86638618_31174427

def rgetattr(obj, attr, *args):
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)
    return functools.reduce(_getattr, [obj] + attr.split('.'))


#here it is made sure that the gradient dont interfere while backpropagation

def do_fit(opt_net, meta_opt, target_cls, target_to_opt, unroll, optim_it, out_mul, should_train=True):
    
    #use the should_train parameter to evaluate the task and not train the
    
    if should_train:
        opt_net.train()
    else:
        opt_net.eval()   #switch for some specific layers/parts of the model that behave differently during training and inference (evaluating) time. For example, Dropouts Layers, BatchNorm Layers etc.
        unroll = 1
    
    target = target_cls(training=should_train)  #it is the dataset, optimizee(target) yields loss
    optimizee = w(target_to_opt())
    n_params = 0
    for name, p in optimizee.all_named_parameters():
        n_params += int(np.prod(p.size()))

    #there are two outputs of LSTM (long term= Cell State C_t, and Hidden State = h_t)
    hidden_states = [w(Variable(torch.zeros(n_params, opt_net.hidden_sz))) for _ in range(2)]
    cell_states = [w(Variable(torch.zeros(n_params, opt_net.hidden_sz))) for _ in range(2)]
    all_losses_ever = []
    if should_train:
        meta_opt.zero_grad()
    
    #all_losses is the loss for LO(sum of all the losses till unroll)
    all_losses = None

    
    
    ###############  keep track of the gradient values at the inference time  ########################

    grad_list_test = []
    ##############################

    for iteration in range(1, optim_it + 1):  #optim_it = number of iterations the optimizee is trained using LO(inner training steps)
        
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
        
        grad_list_each_step = []
        #updates for each parameter of optimizee is output by using the LO
        for name, p in optimizee.all_named_parameters():
            cur_sz = int(np.prod(p.size()))
            gradients = detach_var(p.grad.view(cur_sz, 1))

            
            ###############  keep track of the gradient values at the inference time  ########################

            if should_train==False:
                grad_list_each_step.append(gradients.tolist())

            ############################################
            
            
            updates, new_hidden, new_cell = opt_net(
                gradients,
                [h[offset:offset+cur_sz] for h in hidden_states],
                [c[offset:offset+cur_sz] for c in cell_states]
            )
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
                all_losses.backward()  #updates the grad values for optimizers parameters at unrolling (during test time: unroll =1)
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


        grad_list_test.append(grad_list_each_step) 
    
    #all_losses_ever is the loss that optimizee gives after each iteration
    return [all_losses_ever, optimizee, grad_list_test]

#target_cls ==> loss of optimizee
#target_to_opt ==> neural net optimizee
#n_epochs ==> the number of epochs, LO is trained for, we will use only the opt_net for which epoch the loss was minimum
#n_test ==> number of times the test is done ; test here represents the training of a model (MNIST with remaining dataset)

@cache.cache
def fit_optimizer(target_cls, target_to_opt, preproc=False, unroll=20, optim_it=1000, n_epochs=500, n_tests=100, lr=0.001, out_mul=1.0):
    opt_net = w(Optimizer(preproc=preproc))
    meta_opt = optim.Adam(opt_net.parameters(), lr=lr)
    
    best_net = None
    best_loss = 100000000000000000
    
    for _ in tqdm(range(n_epochs), 'epochs'):
        for _ in tqdm(range(20), 'iterations'):
            do_fit(opt_net, meta_opt, target_cls, target_to_opt, unroll, optim_it, out_mul, should_train=True)
        
        loss = (np.mean([
            np.sum(do_fit(opt_net, meta_opt, target_cls, target_to_opt, unroll, optim_it, out_mul, should_train=False)[0])
            for _ in tqdm(range(n_tests), 'tests')
        ]))
        print(loss)
        if loss < best_loss:
            print(best_loss, loss)
            best_loss = loss
            best_net = copy.deepcopy(opt_net.state_dict())
            
    return best_loss, best_net   #of optimizer


#fit a neural network using an optimizer

@cache.cache
def fit_normal(target_cls, target_to_opt, opt_class, n_tests=100, n_epochs=100, **kwargs):
    results = []
    grad_list = []
    for i in tqdm(range(n_tests), 'tests'):
        target = target_cls(training=False)
        optimizee = w(target_to_opt())
        optimizer = opt_class(optimizee.parameters(), **kwargs)
        total_loss = []
        for _ in range(n_epochs):
            loss = optimizee(target)
            
            total_loss.append(loss.data.cpu().numpy())
            
            optimizer.zero_grad()
            loss.backward()

            grad_list_per_epoch = []
            #get all the stored gradient values using the optimizer object 
            for p in optimizer.param_groups[0]['params']:
                grad_list_per_epoch.append(p.grad.tolist())

            grad_list.append(grad_list_per_epoch)
            #get all the stored gradient values using the model object
            

            optimizer.step()
        results.append(total_loss)
    return [results, optimizee, grad_list]  #of the optimizeee


#incase one needs to find the best learningrate of an optimizer

def find_best_lr_normal(target_cls, target_to_opt, opt_class, **extra_kwargs):
    best_loss = 1000000000000000.0
    best_lr = 0.0
    for lr in tqdm([1.0, 0.3, 0.1, 0.03, 0.01, 0.003, 0.001, 0.0003, 0.0001, 0.00003, 0.00001], 'Learning rates'):
        try:
            loss = best_loss + 1.0
            loss_arr, net = fit_normal(target_cls, target_to_opt, opt_class, lr=lr, **extra_kwargs)
            loss = np.mean([np.sum(s) for s in loss_arr])
        except RuntimeError:
            pass
        if loss < best_loss:
            best_loss = loss
            best_lr = lr
            best_net = net
    return best_loss, best_net, best_lr
