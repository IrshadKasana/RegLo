
#################################################################

#           meta_optimizer file     

#################################################################



from functools import reduce
from operator import mul

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


def initialize_rnn_hidden_state(dim_sum,n_layers,n_params):
    h0 = Variable(torch.zeros(n_layers,n_params,dim_sum),requires_grad=False)
    h0.data = h0.data.cuda()
    return h0

class MetaOptimizerRNN(nn.Module):

    def __init__(self, model, num_layers, input_dim, hidden_size):
        super(MetaOptimizerRNN, self).__init__()
        self.meta_model = model
        self.first_order = nn.RNN(input_dim,hidden_size,num_layers,batch_first=True,bias=False)
        self.outputer = nn.Linear(hidden_size,1,bias=False)
        self.outputer.weight.data.mul_(0.1)
        self.hidden_size = hidden_size

    def parallel(self):
        self.first_order.cuda()
        self.outputer.cuda()

    def reset_lstm(self, num_layers, keep_states=False, model=None, use_cuda=False):
        self.meta_model.reset()
        self.meta_model.copy_params_from(model)
        if keep_states > 0:
            self.h0 = Variable(self.h0.data)
        else:
            self.h0 = initialize_rnn_hidden_state(self.hidden_size, num_layers, self.meta_model.get_flat_params().size(0))

    def forward(self, x):
        output1, hn1 = self.first_order(x, self.h0)
        self.h0 = hn1
        o1 = self.outputer(output1)
        return o1.squeeze()

    def meta_update(self, model_with_grads, loss, eps=None):
        # First we need to create a flat version of parameters and gradients
        grads = []
        for name, module in model_with_grads.named_modules():
            # print(name)
            if len(module._parameters) != 0:
                grads.append(module._parameters['weight'].grad.data.view(-1).unsqueeze(-1))
                try:
                    grads.append(module._parameters['bias'].grad.data.view(-1).unsqueeze(-1))
                except:
                    pass


        flat_params = self.meta_model.get_flat_params()
        flat_grads = torch.cat(grads)
        reqd_grads = flat_grads
        second_moment = (flat_grads**2).mean() + 1e-5
        flat_grads = flat_grads / torch.sqrt(second_moment)
        inputs = Variable(flat_grads.view(-1, 1).unsqueeze(1))
        if eps is None:
            update = self(inputs)
            distance = None
        else:
            state_adv = inputs.detach() + 0.01 * torch.randn(inputs.shape).cuda().detach()
            step_size = eps / 5
            ori_h0 = self.h0

            for _ in range(10):
                state_adv.requires_grad_()
                with torch.enable_grad():
                    perturbed_update = self(state_adv)
                    self.h0 = ori_h0
                    # print(self.h0[0])
                    update = self(inputs)
                    self.h0 = ori_h0
                    # print(self.h0[0][1][1])
                    distance = F.mse_loss(perturbed_update, update)
                    grad = torch.autograd.grad(distance, [state_adv])[0]
                    state_adv = state_adv.detach() + step_size * torch.sign(grad.detach())
                    state_adv = torch.min(torch.max(state_adv, inputs - eps), inputs + eps)

            self.h0 = ori_h0
            perturbed_inputs = state_adv.detach()
            perturbed_update = self(perturbed_inputs)
            self.h0 = ori_h0
            update = self(inputs)
            distance = F.mse_loss(update, perturbed_update)

            
        flat_params = flat_params + update

        self.meta_model.set_flat_params(flat_params)

        # Finally, copy values from the meta model to the normal one.
        self.meta_model.copy_params_to(model_with_grads)
        return self.meta_model.model, distance, update, reqd_grads


# A helper class that keeps track of meta updates
# It's done by replacing parameters with variables and applying updates to
# them.
class MetaModel:

    def __init__(self, model):
        self.model = model


    def reset(self):
        for module in self.model.modules():
            if len(module._parameters) != 0:
                module._parameters['weight'] = Variable(module._parameters['weight'].data)
                try:
                    module._parameters['bias'] = Variable(module._parameters['bias'].data)
                except:
                    pass


    def get_flat_params(self):
        params = []
        for name, module in self.model.named_modules():
            if len(module._parameters) != 0:
                params.append(module._parameters['weight'].view(-1))
                try:
                    params.append(module._parameters['bias'].view(-1))
                except:
                    pass
        return torch.cat(params)


    def set_flat_params(self, flat_params):
        # Restore original shapes
        offset = 0
        for module in self.model.modules():
            if len(module._parameters) != 0:
                weight_shape = module._parameters['weight'].size()
                weight_flat_size = reduce(mul, weight_shape, 1)
                module._parameters['weight'] = flat_params[
                                               offset:offset + weight_flat_size].view(*weight_shape)
                try:
                    bias_shape = module._parameters['bias'].size()
                    bias_flat_size = reduce(mul, bias_shape, 1)
                    module._parameters['bias'] = flat_params[
                                                 offset + weight_flat_size:offset + weight_flat_size + bias_flat_size].view(
                        *bias_shape)
                except:
                    bias_flat_size = 0
                offset += weight_flat_size + bias_flat_size


    def copy_params_from(self, model):
        for modelA, modelB in zip(self.model.parameters(), model.parameters()):
            modelA.data.copy_(modelB.data)

    def copy_params_to(self, model):
        for modelA, modelB in zip(self.model.parameters(), model.parameters()):
            modelB.data.copy_(modelA.data)













#################################################################

#           Train the optimizee model     

#################################################################


import json
import numpy as np
# from meta_optimizer import *
from model import *
from torchvision import datasets, transforms
import argparse
import pickle
import random
import os

parser = argparse.ArgumentParser(description='PyTorch REINFORCE example')
parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                    help='batch size (default: 128)')
parser.add_argument('--seed', type=int, default=1, metavar='N',
                    help='random seed (default: 1)')
parser.add_argument('--optim_steps', type=int, default=1000, metavar='N',
                    help='number of meta optimizer steps (default: 100)')
# parser.add_argument('--truncated_bptt_step', type=int, default=20, metavar='N',
#                     help='step at which it truncates bptt (default: 20)')
# parser.add_argument('--updates_per_epoch', type=int, default=10, metavar='N',
#                     help='updates per epoch (default: 10)')
# parser.add_argument('--max_epoch', type=int, default=200, metavar='N',
#                     help='number of epoch (default: 200)')
parser.add_argument('--hidden_size', type=int, default=20, metavar='N',
                    help='hidden size of the meta optimizer (default: 20)')
parser.add_argument('--num_layers', type=int, default=2, metavar='N',
                    help='number of LSTM layers (default: 1)')
parser.add_argument('--save', type=str, default='/nlsasfs/home/metalearning/nckris/Suraj_LSAIML/l2l/checkpoints_LO/smoothReg',
                   help="location to optimizer checkpoints")
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

random.seed(args.seed)
torch.manual_seed(args.seed)

# transform_train = transforms.Compose([
#     transforms.RandomCrop(32, padding=4),
#     transforms.RandomHorizontalFlip(),
#     transforms.ToTensor(),
#     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
# ])

# transform_test = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
# ])

# trainset = datasets.CIFAR10(root='/home/surajk/Mtech_project/l2l', train=True, download=True, transform=transform_train)
# train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)

# testset = datasets.CIFAR10(root='/home/surajk/Mtech_project/l2l', train=False, download=True, transform=transform_test)
# test_loader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)



transform_test = transforms.Compose([
    transforms.ToTensor()])

trainset = datasets.MNIST(
            "/nlsasfs/home/metalearning/nckris/Suraj_LSAIML/datasets/", train=True, download=True,
            transform=transform_test)
indices = list(range(len(trainset)))
np.random.RandomState(10).shuffle(indices)

indices_test_task = indices[len(indices) // 2:]

test_task_loader_tr = torch.utils.data.DataLoader(trainset, batch_size=100, num_workers=2, 
                                           sampler=torch.utils.data.sampler.SubsetRandomSampler(indices_test_task))

testset = datasets.MNIST(
            "/nlsasfs/home/metalearning/nckris/Suraj_LSAIML/datasets/", train=False, download=True,
            transform=transform_test)

test_task_loader_ts = torch.utils.data.DataLoader(testset, batch_size=100, num_workers=2, shuffle = True)


def accuracy(outputs, labels):  
    _, predicted = torch.max(outputs, 1)
    # print(outputs.size(), predicted.size())
    correct = (predicted == labels).sum().item()
    total = labels.size(0)
    acc = correct / total
    return acc



def test_error(model, test_loader):
    test_loss = 0.0
    actual = []
    with torch.no_grad():
        for _, (image, label) in enumerate(test_loader):
            image, label = image.cuda(), label.cuda()
            f_x = model(image)
            if _ == 0:
                pred=f_x
                actual = label
            else:
                pred = torch.cat([pred, f_x], dim = 0)
                actual = torch.cat([actual,label], dim = 0)

            loss = F.nll_loss(f_x, label)
            test_loss += loss.item()
    test_loss = test_loss / len(test_loader)

    return test_loss, pred, actual


def test():
    save_root = args.save
    meta_model = Model()
    if args.cuda:
        meta_model.cuda()

    meta_optimizer = MetaOptimizerRNN(MetaModel(meta_model), args.num_layers, 1, args.hidden_size)
    if args.cuda:
        meta_optimizer.cuda()

    #load the saved optimizer 
    #     
    optimizer_path = os.path.join(save_root, 'smoothed-optimizer.pt')
    meta_optimizer.load_state_dict(torch.load(optimizer_path))
    model = Model()
    if args.cuda:
        model.cuda()
    train_iter = iter(test_task_loader_tr)
    loss_sum = 0.0
    step = 0

    for param in meta_optimizer.parameters():
        param.requires_grad = False




    updates_list_smoothDM = []
    updates_list_sgd = []
   


    for k in range(args.optim_steps):
        # Keep states for truncated BPTT
        meta_optimizer.reset_lstm(keep_states=k > 0,num_layers=args.num_layers, model=model, use_cuda=args.cuda)
        # hidden = meta_optimizer.module.h0
        # for j in range(args.truncated_bptt_step):
        try:
            x, y = next(train_iter)
        except StopIteration: #if the number of 
            train_iter = iter(test_task_loader_tr)
            x, y = next(train_iter)
        if args.cuda:
            x, y = x.cuda(), y.cuda()

        f_x = model(x)
        loss = F.nll_loss(f_x, y)
        model.zero_grad()
        loss.backward()

        #meta model is the updated model
        meta_model, _ , update, flat_grads = meta_optimizer.meta_update(model, loss.data)
        

        ######################     UPDATE ######################################
        u1 = torch.reshape(update, (-1,1)).cpu().detach().numpy()
        u2 = torch.reshape(flat_grads, (-1,1)).cpu().detach().numpy()
        u_norm = np.dot(np.transpose(u1), u1)[0][0]
        u_norm_sgd = np.dot(np.transpose(u2), u2)[0][0]
        updates_list_smoothDM.append(u_norm.astype(float))
        updates_list_sgd.append(u_norm_sgd.astype(float))

        ######################     UPDATE ######################################


        meta_optimizer.zero_grad()

        f_x = meta_model(x)
        loss = F.nll_loss(f_x, y)

        step += 1
        loss_sum += loss.item()
        avg_loss = loss_sum / step
        print("train loss:", avg_loss)
        # tr_loss.append(loss.item())
        # tr_acc.append(accuracy(f_x, y))
        print("Step {}".format(step))
        # print("Training loss: {}".format(loss.item()))
        # test_loss, pred, label = test_error(meta_model, test_task_loader_ts)
        # test_acc = accuracy(pred, label)
        # print("Testing loss: {}, Testing accuracy: {}".format(test_loss, test_acc))
        print("-----------------------------------\n")
        # ts_loss.append(test_loss)
        # ts_acc.append(test_acc)
        # print(f"train_loss: {tr_loss} , train_acc: {tr_acc}, test_loss: {ts_loss}, test_acc: {ts_acc}")
    print({"updates_lo" : updates_list_smoothDM, "updates_lo_sgd" : updates_list_sgd})
    return {"updates_lo" : updates_list_smoothDM, "updates_lo_sgd" : updates_list_sgd}
if __name__ == '__main__':
    
    hist_list = {}
    for i in range(10):
        print(f"step {i} done")
        hist_list[i] = test()


    file_name  = "/nlsasfs/home/metalearning/nckris/Suraj_LSAIML/l2l/checkpoints_LO/jsons/1update_smooth_dmo_mnist_list.json"
    with open(file_name, "w") as json_file:
        json.dump(hist_list, json_file)