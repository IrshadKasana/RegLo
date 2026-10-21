from pathlib import Path
import sys

from torchvision import datasets, transforms
import torch
import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[2]))

from project_paths import datasets_dir

DATASETS_ROOT = datasets_dir(__file__)




def data_loader(data = 'mnist', batch_size = 128, random_seed = 10):
    
    if data == 'mnist':
        
        transform_test = transforms.Compose([
            transforms.ToTensor(),
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
        return
