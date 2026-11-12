from pathlib import Path
import sys

from torchvision import datasets, transforms
import torch
import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[3]))

from project_paths import datasets_dir


DATASETS_ROOT = datasets_dir(__file__)




def data_loader(data = 'mnist', batch_size = 128, random_seed = 10):
    
    if data == 'mnist':
        
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        trainset = datasets.MNIST(
                    str(DATASETS_ROOT), train=True, download=True,
                    transform=transform_test)
        indices = list(range(len(trainset)))
        np.random.RandomState(random_seed).shuffle(indices)
        indices_train = indices[:len(indices) // 2]

        indices_test = indices[len(indices) // 2:]

        train_loader = torch.utils.data.DataLoader(trainset, batch_size= batch_size, num_workers=2,
                                                sampler=torch.utils.data.sampler.SubsetRandomSampler(indices_train))

        test_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, num_workers=2,
                                                sampler=torch.utils.data.sampler.SubsetRandomSampler(indices_test))


        return train_loader, test_loader

    if data == 'cifar10':
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
        ])

        trainset = datasets.CIFAR10(
                    str(DATASETS_ROOT), train=True, download=True,
                    transform=transform_test)
        indices = list(range(len(trainset)))
        np.random.RandomState(random_seed).shuffle(indices)

        indices_task_train = indices[:len(indices) // 2]

        train_loader = torch.utils.data.DataLoader(trainset, batch_size= batch_size, num_workers=2,
                                                sampler=torch.utils.data.sampler.SubsetRandomSampler(indices_task_train))

        testset = datasets.CIFAR10(
                    str(DATASETS_ROOT), train=False, download=True,
                    transform=transform_test)
        test_loader = torch.utils.data.DataLoader(testset, shuffle=True, batch_size= batch_size, num_workers=2)



        return train_loader, test_loader
    

####################################### 
####################################### while optimizing an optimizee with trained optimizer
#######################################

def data_loader_optimizee(data = 'mnist', batch_size = 128, random_seed = 10):

    if data == 'mnist':

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        trainset = datasets.MNIST(
                    str(DATASETS_ROOT), train=True, download=True,
                    transform=transform_test)
        indices = list(range(len(trainset)))
        np.random.RandomState(random_seed).shuffle(indices)

        indices_task_train = indices[len(indices) // 2:]

        train_loader = torch.utils.data.DataLoader(trainset, batch_size= batch_size, num_workers=2,
                                                sampler=torch.utils.data.sampler.SubsetRandomSampler(indices_task_train))

        testset = datasets.MNIST(
                    str(DATASETS_ROOT), train=False, download=True,
                    transform=transform_test)
        test_loader = torch.utils.data.DataLoader(testset, batch_size= batch_size, shuffle=True, num_workers=2)


        return train_loader, test_loader

    if data == 'cifar10':
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
        ])

        trainset = datasets.CIFAR10(
                    str(DATASETS_ROOT), train=True, download=True,
                    transform=transform_test)

        train_loader = torch.utils.data.DataLoader(trainset, batch_size= batch_size, num_workers=2, shuffle=True)

        testset = datasets.CIFAR10(
                    str(DATASETS_ROOT), train=False, download=True,
                    transform=transform_test)
        test_loader = torch.utils.data.DataLoader(testset, batch_size= batch_size, num_workers=2, shuffle=True)



        return train_loader, test_loader
    
    if data == 'fmnist':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

        trainset = datasets.FashionMNIST(
            str(DATASETS_ROOT), train=True, download=True,
            transform=transform)

        train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

        testset = datasets.FashionMNIST(
            str(DATASETS_ROOT), train=False, download=True,
            transform=transform)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=2)

        return train_loader, test_loader
