import joblib
import sys
from pathlib import Path
import torch.optim as optim
from torchvision import datasets
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import json

from training_func import *
from resnet_model import *

sys.path.append(str(Path(__file__).resolve().parents[1]))

from project_paths import datasets_dir, ensure_dir, repo_root

ROOT = repo_root(__file__)
CHECKPOINTS_DIR = ensure_dir(ROOT / "checkpoints")
DATASETS_DIR = ensure_dir(datasets_dir(__file__))

cache = joblib.Memory(location='_cache', verbose=0)


class Cifar100_Data:
    def __init__(self, batch_size=128):
        dataset_train = datasets.CIFAR100(
            str(DATASETS_DIR), train=True, download=True,
            transform=torchvision.transforms.Compose([ torchvision.transforms.ToTensor()]) #torchvision.transforms.Resize(32),
        )

        self.batch_size= batch_size
        self.loader = torch.utils.data.DataLoader(
            dataset_train, batch_size=self.batch_size, num_workers=4)

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


# for validation

dataset_test = datasets.CIFAR100(
            str(DATASETS_DIR), train=False, download=True,
            transform=torchvision.transforms.Compose([ torchvision.transforms.ToTensor()])) #torchvision.transforms.Resize(32),

val_data_loader = torch.utils.data.DataLoader(
            dataset_test, batch_size=16, num_workers=4)


criterion = nn.CrossEntropyLoss()


def accuracy(outputs, labels):
    _, predicted = torch.max(outputs, 1)
    correct = (predicted == labels).sum().item()
    total = labels.size(0)
    acc = correct / total
    return acc


#loading pretrained LO model (DMO)
DMO_dict = torch.load(CHECKPOINTS_DIR / "CPpoint_LO_lr_0.003.pth")

DMO_instance = w(Optimizer(preproc=True))
DMO_instance.load_state_dict(DMO_dict['optimizer_state_dict'])


# SDMO_dict = torch.load(folder_name+"CPpoint_sDMO_lr_0.003.pth")

# SDMO_instance = w(SDMOptimizer(preproc=True))
# SDMO_instance.load_state_dict(SDMO_dict['optimizer_state_dict'])

history = fit_val_loss_acc(DMO_instance, Cifar100_Data(batch_size=16), ResNet_Model(ResidualBlock, [3, 4, 6, 3], 10), 10)

with open("test_resnet1.json", "w") as json_file:
    json.dump(history, json_file)
