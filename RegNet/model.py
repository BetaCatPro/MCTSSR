import os
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import yaml

from TripleNet.model import SiameseNetwork


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
    def __init__(self, view_data, data_name, category):
        # [labeled_data, unlabeled_data]
        self.labeled_data = view_data[0]
        self.unlabeled_data = view_data[1]
        self.data_name = data_name
        self.category = category

    def __getitem__(self, index):
        lab_data = None
        lab_data_target = None
        unlabeled_data = None
        
        labeled_instance = self.labeled_data[index].reshape(1, -1)
        
        if self.category == 'metric':
            with open('./configs/config.yml', 'r') as file:
                try:
                    config = yaml.safe_load(file)
                except yaml.YAMLError as exc:
                    print(exc)
            in_channels = labeled_instance.shape[1] - 1
            load_path = os.path.join(config['logging_params']['save_dir'], 'view_{}_best_model.pth'.format(self.data_name))
            config['metric_model_params']['in_channels'] = int(in_channels)
            model = SiameseNetwork(**config['metric_model_params'])
            model.load_state_dict(torch.load(load_path, map_location=lambda storage, loc: storage.cuda(0)))
            model.eval()
            
            input_labeled_tensor = torch.reshape(torch.as_tensor(labeled_instance[:,:-1]), (1, -1, in_channels))
            input_repeat_labeled_tensor = input_labeled_tensor.repeat(self.unlabeled_data.shape[0],1,1)
            input_unlabeled_tensor = torch.reshape(torch.as_tensor(self.unlabeled_data[:, :-1]), (self.unlabeled_data.shape[0], -1, in_channels))
            output, _ = model(input_repeat_labeled_tensor, input_unlabeled_tensor, input_unlabeled_tensor)
            r_idx = np.argsort(output.detach().numpy().reshape(1,-1))[:, :2][0]
            
            # list_similarity = []
            # list_trans_unlabeled = []
            # input_labeled_tensor = torch.reshape(torch.as_tensor(labeled_instance[:,:-1]), (1, -1, in_channels))
            # for unlabeled_instance in self.unlabeled_data:
            #     unlabeled = unlabeled_instance.reshape(1,-1)[:,:-1]
            #     with torch.no_grad():
            #         input_unlabeled_tensor = torch.reshape(torch.as_tensor(unlabeled), (1, -1, in_channels))
            #         output, _ = model(input_labeled_tensor, input_unlabeled_tensor, input_unlabeled_tensor)

            #         list_similarity.append(float(output.item()))
            #         list_trans_unlabeled.append(input_unlabeled_tensor)
            # r_idx = np.argsort(list_similarity)[:2]
            # unlabeled_data = [torch.reshape(model(list_trans_unlabeled[i], None, None, True), (1,64)).detach() for i in r_idx]
            
            print('--------- index: {} ----- r_idx: {} -----'.format(index, r_idx))
            unlabeled_data = [torch.reshape(model(input_unlabeled_tensor[i], None, None, True), (1,64)).detach() for i in r_idx]
            lab_data = torch.reshape(model(input_labeled_tensor, None, None, True), (1,64)).detach()
        else:
            r_idx = random.sample(range(0, self.unlabeled_data.shape[0]), 2)
            unlabeled_data = torch.as_tensor(self.unlabeled_data[r_idx][:, :-1]).to(torch.float32)
            lab_data = torch.as_tensor(labeled_instance[:, :-1]).to(torch.float32)

        lab_data_target = torch.as_tensor(labeled_instance[:, -1]).to(torch.float32)
    
        return [lab_data, lab_data_target, unlabeled_data[0], unlabeled_data[1]]

    def __len__(self):
        return len(self.labeled_data)
