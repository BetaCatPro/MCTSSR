import math
import os

import datetime
import pandas as pd
import yaml
from scipy.io import arff

from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from TripleNet.triplet_net import train_triple_net

from create_pairs import generate_file
from ctssr import CTSSR

import warnings

warnings.filterwarnings("ignore")


def split_labeled_and_unlabeled_data(target_data, scale, rs):
    training_data = target_data.sample(2000, random_state=rs)
    labeled_data = training_data.sample(math.ceil(2000 * scale), random_state=rs)
    unlabeled_data = training_data.drop(labeled_data.index)
    test_data = target_data.drop(training_data.index)

    return labeled_data, unlabeled_data, test_data


if __name__ == '__main__':
    with open('configs/config.yml', 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)

    data_dir = os.listdir(config['experiment_params']['base_data_dir'])

    for run_iter in range(config['experiment_params']['run_iter']):
        for scale in config['dataset_params']['scale_labeled_samples']:
            for file_name in data_dir:
                start_time = datetime.datetime.now()

                file = file_name.split('.')[0]
                
                print('{} Exp:{}, Scale: {}, DataSet: {} {}'.format('*'*10, run_iter, scale, file, '*'*10))

                data, meta = arff.loadarff(os.path.join(config['experiment_params']['base_data_dir'], file_name))
                data_labeled = pd.DataFrame(data)
                in_channels = data_labeled.shape[1] - 1
                labeled_data, unlabeled_data, test_data = split_labeled_and_unlabeled_data(data_labeled, scale, run_iter)
                
                
                save_dir = './experiment/metric_ablation/{}/{}'.format(file, scale)
                if config['regression_exp_params']['category'] == 'metric':
                    generate_file(file, labeled_data, unlabeled_data, test_data, scale)
                    # train_triple_net(file, in_channels, scale)
                    
                    save_dir = './experiment/test/{}/{}'.format(file, scale)
                
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                    
                model = CTSSR(
                    scale=scale,
                    file_name=file,
                    run_iter=run_iter
                )
                model.fit(labeled_data, unlabeled_data)
                pred = model.predict(test_data)

                pd_dict = {
                    'experiment_iter': run_iter,
                    'data': file,
                }
                cur_rmse = mean_squared_error(test_data.iloc[:, -1], pred, squared=False)
                cur_mae = mean_absolute_error(test_data.iloc[:, -1], pred)
                cur_r2 = r2_score(test_data.iloc[:, -1], pred)
                pd_dict['rmse'] = cur_rmse
                pd_dict['mae'] = cur_mae
                pd_dict['r2score'] = cur_r2
                
                print('{} DataSet: {}, RMSE: {} {}'.format('*'*10, file, cur_rmse, '*'*10))
                print('{} DataSet: {}, MAE: {} {}'.format('*'*10, file, cur_mae, '*'*10))
                print('{} DataSet: {}, R2: {} {}'.format('*'*10, file, cur_r2, '*'*10))
                    
                pd.DataFrame(pd_dict, index=[0]).to_csv('{}/{}-{}.csv'.format(save_dir,
                                                                              run_iter,
                                                                              datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")),
                                                        index = False)

                end_time = datetime.datetime.now()
                print("耗时: {} 秒".format((end_time - start_time).seconds))
