import os

import yaml
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader

from TripleNet.model import *
from TripleNet.training import *
from utils.tools import to_csv


def get_pre_data_loaders(**config):
    pre_triplet_dataset = PreTripletDataset(os.path.join(config['dataset_params']['dataset_path'], 'labeled_data.csv'),
                                            os.path.join(config['dataset_params']['dataset_path'], 'pairs_train.csv'))
    pre_triplet_loader = DataLoader(pre_triplet_dataset, shuffle=True, num_workers=1,
                                    batch_size=config['metric_exp_params']['batch_size'])
    return pre_triplet_loader


def get_cur_data_loaders(number_pos_neg, **config):
    triplet_dataset = TripletDataset(os.path.join(config['dataset_params']['dataset_path'], 'labeled_data.csv'),
                                     os.path.join(config['dataset_params']['dataset_path'], 'unlabeled_data.csv'),
                                     os.path.join(config['dataset_params']['dataset_path'], 'pairs_triplet_train.csv'),
                                     number_pos_neg
                                     )
    triplet_loader = DataLoader(triplet_dataset, shuffle=True, num_workers=0,
                                batch_size=config['metric_exp_params']['batch_size'])
    return triplet_loader


def train(epochs, data_loader, model, optimizer, device, in_channels, data='', save_path=None):
    list_train_loss = []
    cur_min_val_error = float(1e8)

    for epoch in range(epochs):
        model.train()
        train_loss = train_triplet(data_loader, model, optimizer, device, in_channels)
        list_train_loss.append(train_loss)

        if (epoch + 1) % 10 == 0:
            print('{} Epoch: {} rank loss: {} {}'.format('*'*10, epoch + 1, train_loss, '*'*10))

        if train_loss < cur_min_val_error:
            if save_path:
                torch.save(model.state_dict(), os.path.join(save_path, '{}_best_model.pth').format(data))
            cur_min_val_error = train_loss


def create_pairs(data, target, target_index, model, device):
    pairs = []
    for tup in data.itertuples():
        un_tensor = th.from_numpy(np.asarray(tup[1::])).reshape(1, -1).to(device)
        un_tensor = th.reshape(un_tensor, (un_tensor.shape[0], -1, un_tensor.shape[1]))
        u_sim, _ = model(target, un_tensor, un_tensor)
        pair = (target_index, tup[0], float(u_sim))
        pairs.append(pair)

    return pairs


def generate_sim_pairs_file(model, device, labeled_data, unlabeled_data, save_path):
    pairs_label = []
    labeled_data = pd.read_csv(labeled_data).iloc[:, 1:-1]
    unlabeled_data = pd.read_csv(unlabeled_data).iloc[:, 1:-1]

    # with cluster
    clf_kmeans = KMeans(n_clusters=9)
    y_pred = clf_kmeans.fit(unlabeled_data)
    center = clf_kmeans.cluster_centers_

    for labeled_idx in labeled_data.index.values.tolist():
        input_tensor_1 = labeled_data.loc[labeled_idx]
        input_tensor_1 = th.from_numpy(np.asarray(input_tensor_1)).reshape(1, -1).to(device)
        input_tensor_1 = th.reshape(input_tensor_1, (input_tensor_1.shape[0], -1, input_tensor_1.shape[1]))

        diff_list = []
        for idx in center:
            input_tensor_2 = th.from_numpy(np.asarray(idx)).reshape(1, -1).to(device)
            input_tensor_2 = th.reshape(input_tensor_2, (input_tensor_2.shape[0], -1, input_tensor_2.shape[1]))
            with th.no_grad():
                difference, _ = model(input_tensor_1, input_tensor_2, input_tensor_2)
            diff_list.append(float(difference))

        sorted_res = np.argsort(diff_list)
        sim_cluster_center = sorted_res[0]
        unsim_cluster_center = sorted_res[-1]

        sim_cluster = unlabeled_data.loc[[x for x, y in list(enumerate(y_pred.labels_)) if y == sim_cluster_center]]
        dis_sim_cluster = unlabeled_data.loc[
            [x for x, y in list(enumerate(y_pred.labels_)) if y == unsim_cluster_center]]

        sim_pair = create_pairs(sim_cluster, input_tensor_1, labeled_idx, model, device)
        unsim_pair = create_pairs(dis_sim_cluster, input_tensor_1, labeled_idx, model, device)
        pairs_label.extend(sim_pair)
        pairs_label.extend(unsim_pair)

    pairs_labeled = np.asarray(pairs_label)
    df_pairs_labeled = pd.DataFrame.from_records(
        pairs_labeled, columns=['Sample1', 'Sample2', 'Difference'])
    to_csv(df_pairs_labeled, save_path)


def train_triple_net(data_name, in_channels, scale):
    with open('configs/config.yml', 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)

    config['metric_model_params']['in_channels'] = int(in_channels)

    config['dataset_params']['dataset_path'] = os.path.join(config['dataset_params']['dataset_path_fixed'], data_name,
                                                            str(scale))
    config['dataset_params']['evaluation_path'] = os.path.join(config['dataset_params']['evaluation_path_fixed'], data_name,
                                                               str(scale))

    save_path = os.path.join(config['logging_params']['save_dir'])
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    torch.manual_seed(config['logging_params']['manual_seed'])
    np.random.seed(config['logging_params']['manual_seed'])

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print('=========== training pre_triplet ===========')

    model = SiameseNetwork(**config['metric_model_params']).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['metric_exp_params']['LR'])

    pre_triplet_loader = get_pre_data_loaders(**config)
    # train(config['metric_exp_params']['epochs'], pre_triplet_loader, model, optimizer, device,
    #       config['metric_model_params']['in_channels'], 'view_{}'.format(data_name), save_path)
    
    train(config['metric_exp_params']['epochs'], pre_triplet_loader, model, optimizer, device,
          config['metric_model_params']['in_channels'])

    print('=========== training triplet ===========')

    generate_sim_pairs_file(model,
                            device,
                            os.path.join(config['dataset_params']['dataset_path'], 'labeled_data.csv'),
                            os.path.join(config['dataset_params']['dataset_path'], 'unlabeled_data.csv'),
                            os.path.join(config['dataset_params']['dataset_path'], 'pairs_triplet_train.csv')
                            )

    triplet = SiameseNetwork(**config['metric_model_params']).to(device)
    optimizer = torch.optim.Adam(triplet.parameters(), lr=config['metric_exp_params']['LR'])
    triplet_loader = get_cur_data_loaders(config['metric_exp_params']['number_pos_neg'], **config)

    train(config['metric_exp_params']['epochs'], triplet_loader, triplet, optimizer, device,
          config['metric_model_params']['in_channels'], 'view_{}'.format(data_name), save_path)
