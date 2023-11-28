import copy
import math
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from m5py import M5Prime
from sklearn.ensemble import RandomForestRegressor
from sklearn import neighbors, linear_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.utils import shuffle

import torch as th

import yaml
from RegNet.model import RegNet
from RegNet.reg_net import train_net

from create_pairs import generate_file
from select_samples import select_sim_samples
from transform_data import transform


class S2RM:
    def __init__(self, co_learners, scale, in_channels, iteration=20, tau=1, gr=1, K=3, s_strategy='metric'):
        """
        :param co_learners: 协同回归器
        :param scale: 有标记数据划分比例
        :param in_channels: SimNet 输入通道
        :param iteration: 最大迭代次数
        :param tau: 控制样本选择的动态阈值
        :param gr: grow rate
        :param K: 伪标签编辑策略 ===> 重复 K 次预测，最后的预测结果取平均，对于稳定的样本，其 K 次预测的结果不会发生大变化
        :param s_strategy: 无标记样本选择策略
        """
        self.co_learners = co_learners
        self.scale = scale
        self.in_channels = in_channels
        self.iteration = iteration
        self.tau = tau
        self.gr = gr
        self.K = K
        self.s_strategy = s_strategy

        self.labeled_data = None
        # 袋内数据
        self.labeled_l_data = []
        # 袋外数据
        self.labeled_v_data = []
        # 从 pool 中选择的数据
        self.pi = []
        
        self.file_name = ''

    def _init_training_data_and_reg(self, data_name, labeled_data):
        """
        初始化袋内/袋外数据, 协同回归器
        :param labeled_data: 有标记数据集
        :return: None
        """
        self.labeled_data = labeled_data.reset_index(drop=True)

        for n in range(len(self.co_learners)):
            inner_data = self.labeled_data.sample(math.ceil(self.labeled_data.shape[0] * .8))
            outer_data = self.labeled_data.drop(inner_data.index)
            inner_data = transform(data_name, inner_data, in_channels=self.in_channels)
            outer_data = transform(data_name, outer_data, in_channels=self.in_channels)
            self.labeled_l_data.append(inner_data)
            self.labeled_v_data.append(outer_data)

            self.pi.append(None)

            self.co_learners[n].fit(inner_data[:, :-1], inner_data[:, -1])

    def _co_training(self, data_name, selected):
        """
        协同训练过程
        :param selected: 度量网络挑选的样本集合
        :return: data_size
        """
        origin_labeled_data_size = self.labeled_data.shape[0]

        while not selected.empty:
            self._inner_test(selected)
            for i in range(len(self.co_learners)):
                pool = shuffle(selected).reset_index(drop=True)
                _pi, _index = self._select_relevant_examples(i, pool, self.labeled_v_data[i], self.gr, data_name)
                self.pi[i] = _pi
                selected = pool.drop(_index)

                if selected.empty:
                    break

            if not any([p.size for p in self.pi]):
                break

            for i in range(len(self.co_learners)):
                if self.pi[i].size == 0:
                    continue

                self.labeled_l_data[i] = np.vstack([self.labeled_l_data[i], self.pi[i]])
                current_labeled_data = self.labeled_l_data[i]
                self.co_learners[i].fit(current_labeled_data[:, :-1], current_labeled_data[:, -1])

        now_labeled_data_size = self.labeled_data.shape[0]
        return origin_labeled_data_size, now_labeled_data_size - origin_labeled_data_size

    def _select_relevant_examples(self, j, unlabeled_data, labeled_v_data, gr, data_name):
        """
        选择置信度样本集合
        :param j: 当前学习器索引
        :param unlabeled_data: 无标签样本缓冲池
        :param labeled_v_data: 当前学习器袋外数据
        :param gr: grow rate
        :return: pi: 被选择置信数据
        """
        transform_unlabeled_data = transform(data_name, unlabeled_data, in_channels=self.in_channels)

        delta_x_u_result = []

        labeled_data_j = labeled_v_data[:, :-1]
        labeled_target_j = labeled_v_data[:, -1]

        # 计算当前 learner_j 在 V_j 上的 RMSE
        epsilon_j = mean_squared_error(self.co_learners[j].predict(labeled_data_j), labeled_target_j,
                                       squared=False)

        # 计算其他学习器的预测结果
        others_learner_pred_list = []
        for k in range(self.K):
            for i in range(len(self.co_learners)):
                if i != j:
                    others_learner_pred_list.append(self.co_learners[i].predict(transform_unlabeled_data[:, :-1]))
        std_res = np.std(np.vstack(others_learner_pred_list).reshape(-1, len(others_learner_pred_list)), axis=1)
        stable_samples = [i for i in std_res if i < 0.4]
        # stable_samples = [i for i in std_res]
        stable_samples_idx = [list(std_res).index(i) for i in stable_samples]
        mean_prediction = sum([pred[stable_samples_idx] for pred in others_learner_pred_list]) / len(
            others_learner_pred_list)

        pred_unlabeled_data = np.hstack(
            [transform_unlabeled_data[stable_samples_idx][:, :-1], mean_prediction.reshape(-1, 1)])
        unlabeled_data = unlabeled_data.loc[stable_samples_idx].reset_index(drop=True)
        with_pseudo_label_unlabeled_data = pd.concat(
            [unlabeled_data.iloc[:, :-1], pd.DataFrame(mean_prediction)],
            axis=1)
        with_pseudo_label_unlabeled_data.columns = self.labeled_data.columns

        for x_u in pred_unlabeled_data:
            # 将当前 x_u 添加到 L_j 中, 并训练新的回归器用于计算 epsilon‘
            tmp_l_j = np.vstack([self.labeled_l_data[j], x_u])
            new_learner = copy.deepcopy(self.co_learners[j])
            new_learner.fit(tmp_l_j[:, :-1], tmp_l_j[:, -1])

            tmp_epsilon_j = mean_squared_error(new_learner.predict(labeled_data_j), labeled_target_j,
                                               squared=False)
            # 计算 x_u 置信度
            delta_x_u_result.append((epsilon_j - tmp_epsilon_j) / (epsilon_j + tmp_epsilon_j))

        # 获取 gr 个大于 0 的最大的 delta_x_u
        x_u_index = np.argsort(delta_x_u_result)[::-1]
        i_counts = len([_ for _ in delta_x_u_result if _ > 0])
        i_counts = i_counts if i_counts <= gr else gr

        self.labeled_data = pd.concat([self.labeled_data, with_pseudo_label_unlabeled_data.loc[x_u_index[0:i_counts]]])

        return pred_unlabeled_data[x_u_index[0:i_counts]], [stable_samples_idx[i] for i in x_u_index[0:1]]

    def fit(self, data_name, labeled_data):
        self.file_name = data_name
        self._init_training_data_and_reg(data_name, labeled_data)

        selected, unlabeled_data, is_select_all = select_sim_samples(data_name=data_name,
                                                                     tau=self.tau,
                                                                     in_channels=self.in_channels,
                                                                     scale=self.scale,
                                                                     s_strategy=self.s_strategy
                                                                     )

        for it in range(self.iteration):
            print('------- iter: {}/{} -------'.format(it + 1, self.iteration))
            if selected.empty:
                break

            labeled_data_size, selected_data_size = self._co_training(data_name, selected)

            if selected_data_size <= 3:
                break

            if is_select_all:
                break

            generate_file(
                self.labeled_data,
                unlabeled_data,
                None,
                is_first_split=False,
                scale=self.scale
            )

            self.tau = math.pow(.98, it + 1) * self.tau
            selected, unlabeled_data, is_select_all = select_sim_samples(
                data_name=data_name,
                tau=self.tau,
                in_channels=self.in_channels,
                scale=self.scale,
                data_size=labeled_data_size,
                s_strategy=self.s_strategy
            )
        
        # TODO 训练RegNN
        
        # 合并去重
        st_labeled_data = np.unique(np.vstack(self.labeled_l_data), axis=0)
        r_idx = random.sample(range(0, st_labeled_data.shape[0]), int(st_labeled_data.shape[0]*.8))
        
        training_view = st_labeled_data[r_idx]
        validation_view = np.delete(st_labeled_data, r_idx, axis=0)
        unlabeled_view = transform(self.file_name, unlabeled_data, self.in_channels)
        
        print('--------- start training regnet -----------')
        train_net(training_view, validation_view, unlabeled_view)


    def predict(self, data):
        with open('configs/config.yml', 'r') as file:
            try:
                config = yaml.safe_load(file)
            except yaml.YAMLError as exc:
                print(exc)

        device = 'cuda' if th.cuda.is_available() else 'cpu'
        
        t_data = transform(self.file_name, data, self.in_channels)

        load_path = os.path.join('saves', 'reg_model.pth')
        model = RegNet(**config['regression_model_params']).to(device)
        model.load_state_dict(th.load(load_path, map_location=lambda storage, loc: storage.cuda(0)))
        model.eval()
    
        nn_data = th.as_tensor(t_data[:, :-1]).to(th.float32).to(device)
        pred = model(nn_data).cpu().detach().numpy()
        return pred
    
    def _predict(self, data):
        self.labeled_data = self.labeled_data.reset_index(drop=True)

        trans_unlabeled_data = transform(self.file_name, data, in_channels=self.in_channels)
        trans_unlabeled_data_x = trans_unlabeled_data[:, :-1]

        result = []
        pred = []
        weight = [1 / len(self.co_learners)] * len(self.co_learners)
        for learner, w in zip(self.co_learners, weight):
            pred.append(w * learner.predict(trans_unlabeled_data_x))
            result.append(learner.predict(trans_unlabeled_data_x))
        result.append(sum(pred))
        return result

    def _inner_test(self, selected):
        val_data = pd.read_csv('evaluation_dataset/{}/evaluation.csv'.format(self.scale)).drop(['index'], axis=1)
        val_data_y = val_data.iloc[:, -1]

        result = self._predict(val_data)

        print('------- cur pool size: {}-------'.format(selected.shape[0]))
        print('RF1 : {}'.format(mean_squared_error(result[0], val_data_y, squared=False)))
        print('RF2 : {}'.format(mean_squared_error(result[1], val_data_y, squared=False)))
        print('RF3 : {}'.format(mean_squared_error(result[2], val_data_y, squared=False)))
        print('co-training rmse: {}'.format(mean_squared_error(result[3], val_data_y, squared=False)))
        print('co-training r2score: {}'.format(r2_score(val_data_y, result[3])))
