#################################################################

#           Train the optimizee model     

#################################################################


import json
import numpy as np
import torch
from meta_optimizer import *
from model import *
from torchvision import datasets, transforms
import argparse
import torch.nn.functional as F
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


    ts_loss = []
    ts_acc = []
    tr_loss = []
    tr_acc = []
    MA = 0
    use_sgd_flag = False
    steps = 30
    optimizer = torch.optim.SGD(model.parameters(), lr=1)
    count_sgd = 0
    min_loss = 999999.99
    best_model = Model()


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
        optimizer.zero_grad()

        loss.backward()


        if not use_sgd_flag:
        #meta model is the updated model
            meta_model, _ = meta_optimizer.meta_update(model, loss.data)
            meta_optimizer.zero_grad()
            f_x = meta_model(x)

            #keep the best model in hand when trying to start SGD
            if loss.item() < min_loss:
                min_loss = loss.item()
                for modelA, modelB in zip(best_model.parameters(), meta_model.parameters()):
                    modelA.data.copy_(modelB.data) #copying the model params to best model params

        else:
            optimizer.zero_grad()

            optimizer.step()
            count_sgd += 1
            f_x = model(x)
            if count_sgd > 100:
                break
            
        loss = F.nll_loss(f_x, y)

        loss_sum += loss.item()
        avg_loss = loss_sum / (k+1)
        tr_loss.append(loss.item())
        train_acc = accuracy(f_x, y)
        tr_acc.append(train_acc)
        print("Step {}".format(k))
        # print("Training loss: {}".format(loss.item()))
        test_loss, pred, label = test_error(meta_model, test_task_loader_ts)
        test_acc = accuracy(pred, label)
        # print("Testing loss: {}, Testing accuracy: {}".format(test_loss, test_acc))
        print("-----------------------------------\n")
        ts_loss.append(test_loss)
        ts_acc.append(test_acc)
        print(f"train_loss: {loss.item()} , train_acc: {train_acc}, test_loss: {test_loss}, test_acc: {test_acc}")
        if not use_sgd_flag:
            
            if (k+1>=steps):
                MA = np.mean(np.array(ts_loss[k+1-steps:]))
                print(f"\n Moving Average : {MA} \n")
                if(k+1==steps): #initialization
                    prev_MA = MA
                else:
                    if MA > prev_MA:
                        use_sgd_flag = True
                        print("iteration at which the LO training stopped:", k+1)
                    prev_MA = MA
                    for modelA, modelB in zip(best_model.parameters(), model.parameters()):
                        modelB.data.copy_(modelA.data)
        
    return {"train_loss" : tr_loss, "train_acc": tr_acc, "test_loss" : ts_loss, "test_acc": ts_acc}

if __name__ == '__main__':
    
    hist_list = {}
    for i in range(10):
        print(f"step {i} done")
        hist_list[i] = test()


    file_name  = "/nlsasfs/home/metalearning/nckris/Suraj_LSAIML/l2l/checkpoints_LO/jsons/1hist_smooth_dmo_mnist_sgd_list.json"
    with open(file_name, "w") as json_file:
        json.dump(hist_list, json_file)