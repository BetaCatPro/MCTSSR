import os

import numpy as np
import torch as th
import random

import yaml

from TripleNet.model import SiameseNetwork

np.set_printoptions(precision=3, suppress=True)


def transform(data_name, data):
    with open('./configs/config.yml', 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)

    load_path = os.path.join(config['logging_params']['save_dir'], 'view_{}_best_model.pth'.format(data_name))
    config['metric_model_params']['in_channels'] = int(data.shape[1]-1)
    model = SiameseNetwork(**config['metric_model_params'])
    model.load_state_dict(th.load(load_path, map_location=lambda storage, loc: storage.cuda(0)))
    model.eval()

    input_tensor = th.as_tensor(data[:, :-1])
    transformed_data = model(input_tensor, None, None, True).detach().numpy()
    label = data[:, -1].reshape(-1, 1)

    return np.hstack([transformed_data, label])
