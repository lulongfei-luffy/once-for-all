import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as f
from .data_process import *
from torch.utils.data import DataLoader, Dataset,TensorDataset


class latency_net(torch.nn.Module):
    def __init__(self, para):
        n_feature, n_hid0, n_hid1, n_hid2, n_hid3, output = para[0], para[1], para[2],para[3], para[4], para[5]
        super(latency_net,self).__init__()
        self.hidden0 = torch.nn.Linear(n_feature,n_hid0)
        self.hidden1 = torch.nn.Linear(n_hid0,n_hid1)
        self.hidden2 = torch.nn.Linear(n_hid1, n_hid2)
        self.hidden3 = torch.nn.Linear(n_hid2, n_hid3)
        self.predict = torch.nn.Linear(n_hid3, output)

    def forward(self, x):
        x = f.relu(self.hidden0(x))
        x = f.relu(self.hidden1(x))
        x = f.relu(self.hidden2(x))
        x = f.relu((self.hidden3(x)))
        x = self.predict(x)
        return x





