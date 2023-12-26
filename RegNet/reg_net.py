import copy
import datetime
import os

import yaml
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from RegNet.model import *
from RegNet.training import train_reg_net, validate


def get_data_loaders(view_data, **config):
    train_data_dataset = RegDataset(view_data)
    train_data_loader = DataLoader(train_data_dataset, shuffle=True, num_workers=1, batch_size=config['batch_size'])
    return train_data_loader


def update_ema_variables(model, ema_model, alpha, global_step):
    alpha = max(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)


def init_ts(training_data, device, config):
    student_model = RegNet(**config['regression_model_params']).to(device)
    teacher_model = copy.deepcopy(student_model)

    for param in teacher_model.parameters():
        param.requires_grad_(False)

    optimizer = torch.optim.Adam(student_model.parameters(), lr=config['regression_exp_params']['LR'], weight_decay=.02)

    return [student_model, teacher_model], optimizer


def train_net(training_view, validation_view, unlabeled_view):
    with open('configs/config.yml', 'r') as file:
        try:
            config = yaml.safe_load(file)
            config['regression_model_params']['in_channels'] = training_view.shape[1] - 1
        except yaml.YAMLError as exc:
            print(exc)

    torch.manual_seed(config['logging_params']['manual_seed'])
    np.random.seed(config['logging_params']['manual_seed'])
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    view_model, optimizer = init_ts(training_view, device, config)

    cur_min_val_error = float(1e8)

    for epoch in range(config['regression_exp_params']['epochs']):
        view_model[0].train()

        ts_training_data = get_data_loaders([training_view, unlabeled_view], **config['regression_exp_params'])
        loss = train_reg_net(epoch,
                             ts_training_data,
                             view_model,
                             optimizer,
                             device)
        update_ema_variables(view_model[0], view_model[1], .9999, config['regression_exp_params']['epochs'])

        view_model[0].eval()
        stu_validation_rmse, stu_r2 = validate(validation_view, view_model[0], device)
        tea_validation_rmse, tea_r2 = validate(validation_view, view_model[1], device)

        print('{} stu Epoch: {} train loss: {}, val rmse: {}, r2: {} {}'.format('*'*10, epoch + 1, loss, stu_validation_rmse, stu_r2, '*'*10))
        print('{} tea Epoch: {} train loss: {}, val rmse: {} {}'.format('*'*10, epoch + 1, loss, tea_validation_rmse, '*'*10))

        if stu_validation_rmse < cur_min_val_error:
            torch.save(view_model[0].state_dict(), os.path.join('saves', 'reg_model.pth'))
            cur_min_val_error = stu_validation_rmse
            