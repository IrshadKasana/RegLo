
import torch
import torchvision
from hessian_eigenthings import compute_hessian_eigenthings
from pyhessian import hessian
import os
import functools
from functools import reduce
from operator import mul
import torch.optim as optim
from tqdm import tqdm
from torchvision import datasets
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
import json

USE_CUDA = torch.cuda.is_available()
print(USE_CUDA)

def w(v):
    if USE_CUDA:
        return v.cuda()
    return v

dataset_train = datasets.MNIST(
            "/nlsasfs/home/metalearning/nckris/Suraj_LSAIML/datasets", train=True, download=True,
            transform=torchvision.transforms.Compose([ torchvision.transforms.ToTensor()]) #torchvision.transforms.Resize(32),
        )
dataset_test = datasets.MNIST(
            "/nlsasfs/home/metalearning/nckris/Suraj_LSAIML/datasets", train=False, download=True,
            transform=torchvision.transforms.Compose([ torchvision.transforms.ToTensor()])) #torchvision.transforms.Resize(32),
        
indices = list(range(len(dataset_train)))
np.random.RandomState(10).shuffle(indices)
indices = indices[len(indices) // 2:]

dataloader_tr = torch.utils.data.DataLoader(dataset_train, batch_size=128,
                                         sampler=torch.utils.data.sampler.SubsetRandomSampler(indices), num_workers = 32)

dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=128, num_workers = 32)

class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()
        self.linear1 = nn.Linear(28 * 28, 100)
        self.linear2 = nn.Linear(100, 10)

    def forward(self, inputs):
        x = inputs.view(-1, 28 * 28)
        x = F.sigmoid(self.linear1(x))
        x = self.linear2(x)
        return F.log_softmax(x, dim=-1)

    def get_flat_params(self):
        params = []

        for module in self.children():
            params.append(module._parameters['weight'].view(-1))
            params.append(module._parameters['bias'].view(-1))

        return torch.cat(params)

    def set_flat_params(self, flat_params):
        # Restore original shapes
        offset = 0
        for i, module in enumerate(self.children()):
            weight_shape = module._parameters['weight'].size()
            bias_shape = module._parameters['bias'].size()

            weight_flat_size = reduce(mul, weight_shape, 1)
            bias_flat_size = reduce(mul, bias_shape, 1)

            module._parameters['weight'].data = flat_params[
                offset:offset + weight_flat_size].view(*weight_shape).data
            module._parameters['bias'].data = flat_params[
                offset + weight_flat_size:offset + weight_flat_size + bias_flat_size].view(*bias_shape).data

            offset += weight_flat_size + bias_flat_size



def validate(model, dataloader, loss_function):

    tot_loss = 0
    count = 0
    pred = []
    actual = []
    for data in dataloader:
        inp, out = data
        inp, out = w(inp), w(out)
        pred_out = model(inp)
        pred.append(pred_out)
        actual.append(out)
        loss_ = loss_function(pred_out, out)
        tot_loss += loss_.item()
        count += 1
        # print("pred_out-each data", pred_out.size())
    pred = torch.cat(pred, dim = 0)
    actual = torch.cat(actual, dim = 0)
    # print("pred", pred, pred.size())
    acc = accuracy(pred, actual)
    return tot_loss/count , acc


def accuracy(outputs, labels):
    
    _, predicted = torch.max(outputs, 1)
    # print(outputs.size(), predicted.size())
    correct = (predicted == labels).sum().item()
    total = labels.size(0)
    acc = correct / total
    return acc

def eig_values(best_model):
    num_eigenthings = 100  # compute top 100 eigenvalues/eigenvectors
    loss_e = nn.NLLLoss()
    dataloader_tr = torch.utils.data.DataLoader(dataset_train, batch_size=128,
                                         sampler=torch.utils.data.sampler.SubsetRandomSampler(indices), num_workers = 32)
    eigenvals, eigenvecs = compute_hessian_eigenthings(best_model,dataloader_tr,loss_e, num_eigenthings)
    print(eigenvals)
    return eigenvals.tolist(), eigenvecs.tolist()
    



def train_n_val_sgd(model, optimizer, num_steps = 500):
    best_model = Model()
    best_iter = 0
    eigen_values = {}
    diff = 1e-3
    min_eval_loss = 99999.99
    patience = 50
    training_loss = []
    training_acc = []
    validation_loss = []
    validation_acc = []
    train_iter = iter(dataloader_tr)

    for step in tqdm(range(num_steps), "Iteration:"):

        try:
            x, y = next(train_iter)
        except StopIteration:
            train_iter = iter(dataloader_tr)
            x, y = next(train_iter)
        input, output = w(x), w(y)

        optimizer.zero_grad()
        # first forward-backward pass
        loss_function = nn.NLLLoss()
        tr_pred_out = model(input)
        # print("tr_pred_out", tr_pred_out.size(), "output", output.size())

        loss = loss_function(tr_pred_out, output)
        tr_acc = accuracy(tr_pred_out, output)
        loss.backward()
        optimizer.step()
        
        
        val_loss, val_acc = validate(model, dataloader_test, loss_function)

        training_loss.append(loss.item())
        training_acc.append(tr_acc)
        validation_loss.append(val_loss)
        validation_acc.append(val_acc)
        if (step%10 == 0):
            print(f"train loss: {loss.item()}, train acc: {tr_acc}, valid_loss: {val_loss}, valid_acc: {val_acc}")


        check = min_eval_loss - (loss.item())
        if (check > diff):
            min_eval_loss = loss.item()
            for modelA, modelB in zip(best_model.parameters(), model.parameters()):
                modelA.data.copy_(modelB.data)
            best_iter = step
            tolerance = 0

        else:
            tolerance += 1

            if(tolerance > patience):
                eigen_vals, eigen_vecs = eig_values(best_model)
                eigen_values[best_iter] = eigen_vals
                break

        # if(step == 0):
        #     print(f"At the first step: calculating hessian eigen values for iteration {step}")
        #     eigen_vals, eigen_vecs = eig_values(best_model)
        #     eigen_values[step] = eigen_vals

        if(len(eigen_values.keys()) == 0 and step==num_steps-1):
            print(f"At the last step: calculating hessian eigen values for iteration {step}")
            eigen_vals, eigen_vecs = eig_values(best_model)
            eigen_values[best_iter] = eigen_vals                
            print(f"Stopped at iteration {step}.  \n")

    return {"eigen_values": eigen_values, "train_loss" : training_loss, "train_acc": training_acc, "valid_loss": validation_loss, "valid_acc": validation_acc}




if __name__ == "__main__":
    model = w(Model())
    base_optimizer = optim.SGD(model.parameters(), lr=0.1)  # define an optimizer for the "sharpness-aware" update
    history = train_n_val_sgd(model, base_optimizer, num_steps = 500)
    file_name = "/nlsasfs/home/metalearning/nckris/Suraj_LSAIML/l2l/checkpoints_LO/jsons/SGD_train_mnist.json"
    with open(file_name, 'w') as f:
        json.dump(history, f)