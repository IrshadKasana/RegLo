
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch.optim as optim

import random
from tqdm import tqdm_notebook as tqdm
import copy
import joblib
import torchvision

class RegressionLoss:
    def __init__(self, x, y):
        self.x = torch.randn(10,10)
        
    def get_loss(self, theta):
        return torch.sum((self.W.matmul(theta) - self.y)**2) #elementwise squaring
    
class RegressionOptimizee(nn.Module):
    def __init__(self, theta=None):
        super(RegressionOptimizee, self).__init__()
        if theta is None:
            self.theta = nn.Parameter(torch.zeros(10).cuda())
        else:
            self.theta = theta
        
    def forward(self, target):
        return target.get_loss(self.theta)



