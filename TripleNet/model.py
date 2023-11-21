import numpy as np
import pandas as pd
import torch as th
import torch.nn as nn
import torch.nn.functional as F


class SiameseNetwork(nn.Module):
    def __init__(self, **config):
        super(SiameseNetwork, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(config['in_channels'], config['number_of_neurons_1']),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(config['number_of_neurons_1'], config['number_of_neurons_2'])
        )

        self.fc2 = nn.Sequential(
            nn.Linear(config['number_of_neurons_2'] * 2, 1))

    def forward_once(self, x):
        output = self.fc1(x)
        return output

    def forward_second(self, x):
        return self.fc2(x)

    def forward(self, input1, input2, input3, get_trans_data=False):
        if not get_trans_data:
            output1 = self.forward_once(input1.float())
            output2 = self.forward_once(input2.float())
            output3 = self.forward_once(input3.float())
            concat_ap = th.cat((output1, output2), 2)
            concat_an = th.cat((output1, output3), 2)
            output_p = self.forward_second(concat_ap)
            output_n = self.forward_second(concat_an)
            return output_p.float(), output_n.float()
        else:
            output = self.forward_once(input1.float())
            return output


class PreTripletDataset:
    def __init__(self, training_labeled_path, training_pairs_path):
        self.labeled_data = pd.read_csv(training_labeled_path)
        self.pairs = pd.read_csv(training_pairs_path)

    def __getitem__(self, index):
        target_label = self.labeled_data.columns[-1]
        labeled_data = self.labeled_data.drop(columns=[target_label])
        labeled_sample = labeled_data.iloc[index]
        labeled_sample_idx = labeled_data['index'][index]

        element_list = self.pairs[
            (self.pairs['Sample1'] == labeled_sample_idx) | (self.pairs['Sample2'] == labeled_sample_idx)]
        sorted_element_list = element_list.sort_values(by='Difference', ascending=False)

        indices_pos = sorted_element_list.iloc[0]['Sample2'] if sorted_element_list.iloc[0][
                                                                    'Sample1'] == labeled_sample_idx else \
            sorted_element_list.iloc[0]['Sample1']
        indices_neg = sorted_element_list.iloc[-1]['Sample2'] if sorted_element_list.iloc[-1][
                                                                     'Sample1'] == labeled_sample_idx else \
            sorted_element_list.iloc[-1]['Sample1']

        anchor = np.delete(labeled_sample.to_numpy().astype(float), 0)
        positive = np.delete(labeled_data[labeled_data['index'] == indices_pos].to_numpy().astype(float), 0)
        negative = np.delete(labeled_data[labeled_data['index'] == indices_neg].to_numpy().astype(float), 0)

        return anchor, positive, negative

    def __len__(self):
        return len(self.labeled_data)


class TripletDataset:
    def __init__(self, training_labeled, training_unlabeled, pairs_triplet, number_pos_neg):
        self.labeled_data = pd.read_csv(training_labeled)
        self.unlabeled_data = pd.read_csv(training_unlabeled)
        self.pairs_triplet = pd.read_csv(pairs_triplet)
        self.number_pos_neg = number_pos_neg

    def __getitem__(self, index):
        labeled_data = self.labeled_data.iloc[:, 1:-1]
        unlabeled_data = self.unlabeled_data.iloc[:, 1:-1]

        cur_labeled_instance = labeled_data.loc[index]

        unlabeled_idx_list = list(self.pairs_triplet[self.pairs_triplet['Sample1'] == index]['Sample2'])
        sort_idx = np.argsort(self.pairs_triplet[self.pairs_triplet['Sample1'] == index]['Difference'])
        indices_pos = [unlabeled_idx_list[i] for i in sort_idx[: self.number_pos_neg]]
        indices_neg = [unlabeled_idx_list[i] for i in sort_idx[-self.number_pos_neg:]]

        anchor = np.tile(cur_labeled_instance.to_numpy().astype(float), (self.number_pos_neg, 1))
        positive = unlabeled_data.loc[indices_pos].to_numpy().astype(float)
        negative = unlabeled_data.loc[indices_neg].to_numpy().astype(float)

        return anchor, positive, negative

    def __len__(self):
        return len(self.labeled_data)
