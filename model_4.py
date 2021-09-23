import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import random
import pandas as pd
from sklearn.metrics import roc_auc_score
import torch.nn.functional as F
from torchsummary import summary
class ResidualBlock(nn.Module):
    def __init__(self, channels, k, s,p):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(channels, channels, k, stride=s, padding=p)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, k, stride=s, padding=p)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        y = F.relu(self.bn1(self.conv1(x)))
        return self.bn2(self.conv2(y)) + x

class Expert(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 64, 9, 2, 4),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.1),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, 7, 2, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.1),
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 256, 5, 2, 2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.1),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 512, 3, 2, 1),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.BatchNorm2d(512),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.BatchNorm2d(512),

        )

    def forward(self, x):
        out = self.model(x).view(x.shape[0],-1)

        return out
class Tower(nn.Module):
    def __init__(self):
        super(Tower, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 1),
            nn.ReLU(),
            # nn.Sigmoid()
        )
    def forward(self, x):


        return self.model(x)
class MMOE(nn.Module):
    def __init__(self):
        super(MMOE, self).__init__()

        self.experts = nn.ModuleList([Expert() for i in range(6)])
        self.w_gates = nn.ParameterList([nn.Parameter(torch.randn(256*256,6), requires_grad=True) for i in range(4)])
        self.towers = nn.ModuleList([Tower() for i in range(4)])
        self.softmax = nn.Softmax(dim=1)
    def forward(self, x):

        experts_o = [e(x) for e in self.experts]
        experts_o_tensor = torch.stack(experts_o)
        gates_o = [self.softmax(x.view(x.shape[0],-1) @ g) for g in self.w_gates]

        tower_input = [g.t().unsqueeze(2).expand(-1, -1, 2048)*experts_o_tensor for g in gates_o]
        tower_input = [torch.sum(ti, dim=0) for ti in tower_input]
        tower_input = torch.stack(tower_input)
        final_output = [t(ti) for t, ti in zip(self.towers, tower_input)]
        k = final_output[0]
        m = final_output[1]
        vp = final_output[2]
        vs = final_output[3]


        return k,m,vp,vs

net = MMOE().cuda()
summary(net,(1,256,256))
