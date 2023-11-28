import math
import os

import torch as th
import yaml

from RegNet.reg_net import train_net, RegNet
from TripleNet.model import SiameseNetwork
from TripleNet.triplet_net import train_triple_net
from transform_data import transform


class CTSSR:
    def __init__(self, scale, file_name, run_iter):
        self.scale = scale
        self.file_name = file_name
        self.run_iter = run_iter

        self.labeled_data = None
        self.unlabeled_data = None
        # 训练集
        self.training_data = None
        # 验证集
        self.validation_data = None

    def fit(self, labeled_data, unlabeled_data):
        self.labeled_data = labeled_data.reset_index(drop=True)
        self.unlabeled_data = unlabeled_data.reset_index(drop=True)

        # Train/Validate = 8/2
        self.training_data = self.labeled_data.sample(math.ceil(self.labeled_data.shape[0] * .8))
        self.validation_data = self.labeled_data.drop(self.training_data.index)

        training_view = self.training_data.to_numpy()
        unlabeled_view = self.unlabeled_data.to_numpy()
        validation_view = self.validation_data.to_numpy()
           
        print('=========== Training Regression Model ===========')

        train_net(self.file_name, self.scale, self.run_iter, training_view, unlabeled_view, validation_view)

    def predict(self, data):
        with open('configs/config.yml', 'r') as file:
            try:
                config = yaml.safe_load(file)
            except yaml.YAMLError as exc:
                print(exc)

        device = 'cuda' if th.cuda.is_available() else 'cpu'
        
        t_data = transform(self.file_name, data.to_numpy())
        
        # 非度量策略
        if config['regression_exp_params']['category'] != 'metric':
            config['regression_model_params']['in_channels'] = data.shape[1]-1
            t_data = data.to_numpy()

        load_path = os.path.join('saves', 'reg_model.pth')
        model = RegNet(**config['regression_model_params']).to(device)
        model.load_state_dict(th.load(load_path, map_location=lambda storage, loc: storage.cuda(0)))
        model.eval()
    
        nn_data = th.as_tensor(t_data[:, :-1]).to(th.float32).to(device)
        pred = model(nn_data).cpu().detach().numpy()
        return pred
