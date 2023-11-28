import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class RegNet(nn.Module):
    def __init__(self, **config):
        super(RegNet, self).__init__()
        self.input_layer = nn.Linear(config['in_channels'], config['number_of_neurons_1'])
        self.hidden1 = nn.Linear(config['number_of_neurons_1'], config['number_of_neurons_2'])
        self.hidden2 = nn.Linear(config['number_of_neurons_2'], config['number_of_neurons_3'])
        self.predict = nn.Linear(config['number_of_neurons_3'], 1)
        self.dropout1 = nn.Dropout(p=0.2)
        self.dropout2 = nn.Dropout(p=0.3)
        self.batch_normal1 = nn.BatchNorm1d(config['number_of_neurons_2'])
        self.batch_normal2 = nn.BatchNorm1d(config['number_of_neurons_3'])

    def forward(self, input):
        out = self.input_layer(input)
        out = F.relu(out)
        
        out = self.hidden1(out)
        # out = self.batch_normal1(out)
        out = self.dropout1(out)
        out = F.relu(out)

        out = self.hidden2(out)
        # out = self.batch_normal2(out)
        out = self.dropout2(out)
        out = F.relu(out)

        out = self.predict(out)
        return out


class RegDataset:
    def __init__(self, view_data):
        # view_data => [labeled_data, unlabeled_data]
        self.labeled_data = view_data[0]
        self.unlabeled_data = view_data[1]

    def __getitem__(self, index):
        lab_data = torch.as_tensor(self.labeled_data[index][:-1]).to(torch.float32)
        lab_data_target = torch.as_tensor(self.labeled_data[index][-1]).to(torch.float32)

        r_idx = random.sample(range(0, self.unlabeled_data.shape[0]), 2)
        lab_unlabeled_data = torch.as_tensor(self.unlabeled_data[r_idx][:, :-1]).to(torch.float32)

        return [lab_data, lab_data_target, lab_unlabeled_data[0], lab_unlabeled_data[1]]

    def __len__(self):
        return len(self.labeled_data)
