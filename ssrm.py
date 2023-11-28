import copy
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from m5py import M5Prime
from sklearn.ensemble import RandomForestRegressor
from sklearn import neighbors, linear_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.utils import shuffle

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

    def _init_training_data_and_reg(self, data_name, labeled_data):
        """
        初始化袋内/袋外数据, 协同回归器
        :param labeled_data: 有标记数据集
        :return: None
        """
        self.labeled_data = labeled_data.reset_index(drop=True)

        #### test start
        # origin_data = self.labeled_data
        # transform_data = transform(data_name, origin_data, in_channels=self.in_channels)
        # plt.axis('off')
        # plt.scatter(origin_data.iloc[:, 0], origin_data.iloc[:, 1])
        # plt.savefig('origin_L.svg', dpi=300)
        # plt.show()
        # plt.clf()
        #
        # plt.axis('off')
        # plt.scatter(transform_data[:, 0], transform_data[:, 1])
        # plt.savefig('transform_L.svg', dpi=300)
        # plt.show()
        # plt.clf()

        # poly_reg = PolynomialFeatures(degree=2)
        # x_poly = poly_reg.fit_transform(transform_data[:, 0].reshape(-1, 1))
        #
        # learner = linear_model.LinearRegression()
        # learner.fit(x_poly, transform_data[:, 1].reshape(-1, 1))
        # pred = learner.predict(x_poly)
        #
        # plt.axis('off')
        # plt.scatter(transform_data[:, 0], transform_data[:, 1])
        # plt.plot(transform_data[:, 0], pred, c='r')
        # plt.savefig('regressor1.svg', dpi=300)
        # plt.show()
        #
        # plt.axis('off')
        # plt.scatter(transform_data[10:-100, 0], transform_data[10:-100, 1])
        # plt.plot(transform_data[10:-100, 0], pred[10:-100], c='g')
        # plt.savefig('regressor2.svg', dpi=300)
        # plt.show()
        #
        # plt.axis('off')
        # plt.scatter(transform_data[100:-10, 0], transform_data[100:-10, 1])
        # plt.plot(transform_data[100:-10, 0], pred[100:-10], c='y')
        # plt.savefig('regressor3.svg', dpi=300)
        # plt.show()

        """
        origin_data = self.labeled_data
        transform_data = transform(data_name, origin_data, in_channels=self.in_channels)
        plt.axis('off')
        plt.scatter(origin_data.iloc[:, 0], origin_data.iloc[:, 1], c='g')
        plt.savefig('origin_U.svg', dpi=300)
        plt.show()
        plt.clf()

        plt.axis('off')
        plt.scatter(transform_data[:, 0], transform_data[:, 1], c='g')
        plt.savefig('transform_U.svg', dpi=300)
        plt.show()
        plt.clf()
        #### test stop
        """

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
            self._inner_test(data_name, selected)
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

    def predict(self, data_name, data, methods=None):
        """
        methods: ['co_train', 'm5p'] ===>
        co_train : 采用协同预测
        m5p : 利用带伪标签数据训练的 m5p 模型
        :param data_name: 数据文件名
        :param methods: 数据预测的预测方法
        :param data: 待预测样本
        :return: result: 预测结果
        """
        self.labeled_data = self.labeled_data.reset_index(drop=True)
        train_data_x = self.labeled_data.iloc[:, :-1]
        train_data_y = self.labeled_data.iloc[:, -1]

        unlabeled_data = data
        trans_unlabeled_data = transform(data_name, data, in_channels=self.in_channels)
        trans_unlabeled_data_x = trans_unlabeled_data[:, :-1]
        unlabeled_data_x = unlabeled_data.iloc[:, :-1]

        trans_labeled_data = transform(data_name, self.labeled_data, in_channels=self.in_channels)
        trans_labeled_data_x = trans_labeled_data[:, :-1]
        trans_labeled_data_y = trans_labeled_data[:, -1]

        result = []
        if methods is None:
            methods = ['co_train']
        if 'co_train' in methods:
            pred = []
            weight = [1 / len(self.co_learners)] * len(self.co_learners)
            for learner, w in zip(self.co_learners, weight):
                pred.append(w * learner.predict(trans_unlabeled_data_x))
                result.append(learner.predict(trans_unlabeled_data_x))
            result.append(sum(pred))
        if 'm5p' in methods:
            m5 = M5Prime(min_samples_leaf=4)
            m5.fit(trans_labeled_data_x, trans_labeled_data_y)
            result.append(m5.predict(trans_unlabeled_data_x))

        return result

    def _inner_test(self, data_name, selected):
        val_data = pd.read_csv('evaluation_dataset/{}/evaluation.csv'.format(self.scale)).drop(['index'], axis=1)
        val_data_y = val_data.iloc[:, -1]

        result = self.predict(data_name, val_data, methods=['co_train'])

        print('------- cur pool size: {}-------'.format(selected.shape[0]))
        print('RF1 : {}'.format(mean_squared_error(result[0], val_data_y, squared=False)))
        print('RF2 : {}'.format(mean_squared_error(result[1], val_data_y, squared=False)))
        print('RF3 : {}'.format(mean_squared_error(result[2], val_data_y, squared=False)))
        print('co-training rmse: {}'.format(mean_squared_error(result[3], val_data_y, squared=False)))
        print('co-training r2score: {}'.format(r2_score(val_data_y, result[3])))
        # print('m5p : {}'.format(mean_squared_error(result[4], val_data_y, squared=False)))
