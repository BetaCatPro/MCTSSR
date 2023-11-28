from sklearn.metrics import r2_score
import torch
from torch.nn import MSELoss, SmoothL1Loss
import numpy as np


def sigmoid_ramp_up(current, ramp_up_length=30):
    """
    Exponential ramp-up from https://arxiv.org/abs/1610.02242
    """
    if ramp_up_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, ramp_up_length)
        phase = 1.0 - current / ramp_up_length
        return np.exp(-5.0 * phase * phase).astype(np.float32)


def get_mix_data(stu, tea, m_data1, m_data2, lam):
    un_target1 = tea(m_data1)
    un_target2 = tea(m_data2)
    mix_target = lam * un_target1 + (1 - lam) * un_target2
    mix_data = lam * m_data1 + (1 - lam) * m_data2
    mix_data_target = stu(mix_data)

    return mix_target, mix_data_target


def train_reg_net(epoch, data_loader, view_model, optimizer, device):
    loss = 0
    train_loss = 0
    smooth_l1_loss = SmoothL1Loss()
    # smooth_l1_loss = MSELoss()

    for i, data in enumerate(data_loader, 0):
        
        alpha = 0.2
        lam = np.random.beta(alpha, alpha)

        lab_view = [da.to(device) for da in data]
        [lab_data, lab_data_target, lab_unlabeled_data1, lab_unlabeled_data2] = lab_view

        optimizer.zero_grad()

        # supervised loss
        model_supervised_loss = smooth_l1_loss(view_model[0](lab_data), lab_data_target)
        mix_target, mix_data_target = get_mix_data(view_model[0],
                                                    view_model[0],
                                                    lab_unlabeled_data1,
                                                    lab_unlabeled_data2,
                                                    lam)
        # 一致性正则
        inner_consistency_loss = smooth_l1_loss(mix_target, mix_data_target)
        # total loss
        alpha_weight = 1.0 * sigmoid_ramp_up(epoch + 1)
        loss = model_supervised_loss + alpha_weight * inner_consistency_loss
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    return np.sqrt(train_loss/len(data_loader))


def validate(data, model, device):
    mse_loss = MSELoss()
    with torch.no_grad():
        data = torch.as_tensor(data).to(torch.float32).to(device)
        val_data, targets = data[:, :-1], data[:, -1]
        outputs = model(val_data)
        mse = mse_loss(outputs, targets)
        r2 = r2_score(targets.cpu(), outputs.cpu())
    return np.sqrt(mse.item()), r2
