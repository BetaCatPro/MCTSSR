import datetime
import os

import yaml
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from RegNet.model import *
from RegNet.training import train_reg_net, validate
from transform_data import transform


def get_data_loaders(view_data, file_name, **config):
    train_data_dataset = RegDataset(view_data, file_name, category=config['category'])
    train_data_loader = DataLoader(train_data_dataset, shuffle=True, num_workers=1, batch_size=config['batch_size'])
    return train_data_loader


def train_net(file_name, scale, run_iter, training_view, unlabeled_view, validation_view):
    with open('configs/config.yml', 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)

    torch.manual_seed(config['logging_params']['manual_seed'])
    np.random.seed(config['logging_params']['manual_seed'])
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    validation = transform(file_name, validation_view)

    # 非度量策略
    if config['regression_exp_params']['category'] != 'metric':
        config['regression_model_params']['in_channels'] = training_view.shape[1]-1
        validation = validation_view
        
    reg_model = RegNet(**config['regression_model_params']).to(device)
    optimizer = torch.optim.Adam(reg_model.parameters(), lr=config['regression_exp_params']['LR'], weight_decay=.02)

    cur_min_val_error = float(1e8)
    train_loss = []
    val_loss = []

    for epoch in range(config['regression_exp_params']['epochs']):
        reg_model.train()

        training_data = get_data_loaders([training_view, unlabeled_view], file_name, **config['regression_exp_params'])
        loss = train_reg_net(epoch,
                             training_data,
                             reg_model,
                             optimizer,
                             device,
                             config['regression_exp_params']['epochs'])
        train_loss.append(float(loss))

        reg_model.eval()
        validation_rmse = validate(validation, reg_model, device)
        val_loss.append(validation_rmse)

        print('{} Epoch: {} training loss: {}, val rmse: {} {}'.format('*'*10, epoch + 1, loss, validation_rmse, '*'*10))

        if validation_rmse < cur_min_val_error:
            torch.save(reg_model.state_dict(), os.path.join('saves', 'reg_model.pth'))
            cur_min_val_error = validation_rmse

    save_dir = './experiment/training/{}/{}'.format(file_name, scale)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    pd_dict = {
        'training_loss': train_loss,
        'val_rmse': val_loss
    }
    pd.DataFrame(pd_dict).to_csv('{}/{}-{}.csv'.format(save_dir, run_iter,
                                                       datetime.datetime.now().strftime(
                                                           "%Y-%m-%d-%H-%M-%S")))
    plot_res(config['regression_exp_params']['epochs'], pd_dict)


def plot_res(epochs, data):
    plt.rc('font', family='Times New Roman')
    plt.grid(True, linestyle=':', color='gray', alpha=0.6)

    plt.plot(range(epochs), data.get('training_loss'), 'r-', markerfacecolor='r', label='consistency loss')
    plt.plot(range(epochs), data.get('val_rmse'), 'b-', markerfacecolor='b', label='val loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.show()
    plt.clf()