#!/usr/bin/env python
# coding: utf-8

# name = "k-PPCAs"
# version = "0.0.1"
# authors = [
#   {name="Ke Han", email="kh19r@fsu.edu"},
#   {name="Adrian Barbu", email="abarbu@fsu.edu"},
# ]

import numpy as np
import torch
from scipy.io import loadmat
from scipy.io import savemat

import os

# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import math
from collections import OrderedDict
from torch.utils.data import DataLoader, Dataset

from sklearn import metrics


# Dataset
class MixtureDataset(Dataset):
    def __init__(self, values, labels, local_id=None):
        self.values = values
        self.labels = labels
        self.len = len(self.labels)
        self.original_id = [i for i in range(self.len)]
        self.local_id = local_id
        self.d = self.values.shape[1]

        self.dict = OrderedDict.fromkeys(set(self.labels))
        for i in range(len(self.values)):
            if self.dict[self.labels[i]] is None:
                self.dict[self.labels[i]] = [self.values[i]]
            else:
                self.dict[self.labels[i]].append(self.values[i])
        for i in self.dict.keys():
            self.dict[i] = torch.stack(self.dict[i])

    def __getitem__(self, index):
        values = self.values[index]
        labels = self.labels[index]
        original_id = self.original_id[index]
        return values, labels, original_id

    def __len__(self):
        return self.len


def IMAGENET_Feature(feature_address, cls_range=[0, 100]):
    folder_label_map = {}
    y_train = []
    y_val = []
    x_train = torch.empty((0,))
    x_val = torch.empty((0,))
    N_tr = 0
    N_val = 0
    for path in feature_address:
        for sub_path in os.listdir(path):
            path_datafolder = os.path.join(path, sub_path)
            if sub_path == 'train':
                # build a folder-label map first
                classes_list = os.listdir(path_datafolder)
                for global_class_id, filename in enumerate(classes_list):
                    folder_label_map[filename] = global_class_id

                classid = 0
                local_id_trainset = []
                for local_class_id, filename in enumerate(classes_list[cls_range[0]:cls_range[1]]):
                    if classid % 100 == 0:
                        print(classid)
                    path_1datafile = os.path.join(path_datafolder, filename)
                    data1 = loadmat(path_1datafile)
                    x_train = torch.cat(
                        (x_train, torch.tensor(data1['feature'], dtype=torch.float).flatten(start_dim=1)), dim=0)
                    n, dim = data1['feature'].shape[:2]
                    N_tr += n
                    local_id_1cls = [localid for localid in range(n)]
                    local_id_trainset.extend(local_id_1cls)
                    cls = cls_range[0] + local_class_id
                    y_train.extend(list(np.repeat(cls, n)))  # for mat does not have label
                    classid += 1

            if sub_path == 'val':
                classid = 0
                classes_list = os.listdir(path_datafolder)
                for local_class_id, filename in enumerate(classes_list[:cls_range[1]]):
                    if classid % 200 == 0:
                        print(classid)
                    path_1datafile = os.path.join(path_datafolder, filename)
                    data1 = loadmat(path_1datafile)
                    x_val = torch.cat((x_val, torch.tensor(data1['feature']).flatten(start_dim=1)), dim=0)
                    n, dim = data1['feature'].shape[:2]
                    N_val += n
                    y_val.extend(list(np.repeat(folder_label_map[filename], n)))
                    classid += 1

    train_set = MixtureDataset(x_train, y_train, local_id_trainset)
    val_set = MixtureDataset(x_val, y_val)
    return train_set, val_set, folder_label_map

def CUB200_Feature(feature_address, cls_range=[0, 100]):
    folder_label_map = {}
    y_train = []
    y_val = []
    x_train = torch.empty((0,))
    x_val = torch.empty((0,))
    N_tr = 0
    N_val = 0
    for path in feature_address:
        for sub_path in os.listdir(path):
            path_datafolder = os.path.join(path, sub_path)
            if sub_path == 'train':
                # build a folder-label map first
                classes_list=os.listdir(path_datafolder)
                for global_class_id, filename in enumerate(classes_list):
                    folder_label_map[filename]=global_class_id


                classid = 0
                local_id_trainset = []
                for local_class_id, filename in enumerate(classes_list[cls_range[0]:cls_range[1]]):
                    if classid % 100 == 0:
                        print(classid)
                    path_1datafile = os.path.join(path_datafolder, filename)
                    data1 = loadmat(path_1datafile)
                    n, dim = data1['feature'].shape[:2]
                    x_train = torch.cat((x_train, torch.tensor(data1['feature'], dtype=torch.float).flatten(start_dim=1)), dim=0)
                    N_tr += n
                    local_id_1cls = [localid for localid in range(n)]
                    local_id_trainset.extend(local_id_1cls)

                    # y_train.extend(list(np.repeat(data1['label'], n))) # for mat has label
                    # file_label_map[file] = data1['label'].item()
                    cls=cls_range[0]+local_class_id
                    y_train.extend(list(np.repeat(cls, n)))  # for mat does not have label
                    classid += 1

            if sub_path == 'val':
                classid = 0
                classes_list = os.listdir(path_datafolder)
                for local_class_id, filename in enumerate(classes_list[:cls_range[1]]):
                    if classid % 200 == 0:
                        print(classid)
                    path_1datafile = os.path.join(path_datafolder, filename)
                    data1 = loadmat(path_1datafile)
                    n, dim = data1['feature'].shape[:2]
                    x_val = torch.cat((x_val, torch.tensor(data1['feature']).flatten(start_dim=1)), dim=0)
                    N_val += n
                    y_val.extend(list(np.repeat(folder_label_map[filename], n)))
                    classid += 1

    train_set = MixtureDataset(x_train, y_train, local_id_trainset)
    val_set = MixtureDataset(x_val, y_val)
    return train_set, val_set, folder_label_map


def CIFAR100_Feature(feature_address, cls_range=[0, 100]):
    N_lab = 0
    N_unlab = 0
    start_cls, end_cls = cls_range
    for path in feature_address:
        for sub_path in os.listdir(path):
            path_data = os.path.join(path, sub_path)
            if 'train' in sub_path:
                data1 = loadmat(path_data)
                x_train = torch.from_numpy(data1['feature']).flatten(start_dim=1)
                n, dim = x_train.shape
                N_lab += n
                y_train = data1['label'].flatten()
                x_train = x_train[(start_cls <= y_train) * (y_train < end_cls)]
                y_train = y_train[(start_cls <= y_train) * (y_train < end_cls)].tolist()
            if 'val' in sub_path:
                data1 = loadmat(path_data)
                x_val = torch.from_numpy(data1['feature']).flatten(start_dim=1)
                n, dim = x_val.shape
                N_unlab += n
                y_val = data1['label'].flatten()
                x_val = x_val[y_val < end_cls]
                y_val = y_val[y_val < end_cls].tolist()

    train_set = MixtureDataset(x_train, y_train)
    val_set = MixtureDataset(x_val, y_val)
    return train_set, val_set


def combine_augmented_features(feature_address, n_features=640):
    file_label_map = {}
    y = []
    x = torch.empty((0, n_features))
    N_lab = 0
    for path in feature_address:
        path_datafolder = path
        pathid = 0
        for cls, file in enumerate(os.listdir(path_datafolder)):
            if pathid % 10 == 0:
                print(pathid)
            pathid += 1
            path_1datafile = os.path.join(path_datafolder, file)
            data1 = loadmat(path_1datafile)
            x = torch.cat((x,
                           torch.tensor(data1['feature'], dtype=torch.float).flatten(start_dim=1)),
                          dim=0)
            n, dim = data1['feature'].shape[:2]
            N_lab += n
            y.extend(list(np.repeat(data1['label'], n)))
            file_label_map[file] = data1['label']

    train_aug_set = MixtureDataset(x, y)
    return train_aug_set


# kPPCA functions
def logdet_cov(N, S_cls, d, lda):
    k = len(S_cls)
    diag = lda * torch.eye(d)
    diag[:k, :k] = torch.diag(S_cls ** 2 / N + lda)  # commented when k=0
    logdetcov = torch.logdet(diag)
    return logdetcov.clone()

def deltaDiag(N, L_cls, S_cls, lda):
    d2 = S_cls ** 2 / N
    diagM = torch.diag(d2 / (lda * (d2 + lda)))  # k x k
    delta = L_cls.t() @ diagM @ L_cls  # d x d
    return delta.clone()


def score(x, mu_cls, delta_cls, lda, t=1):
    # x: n x d
    # mu_cls: d
    # delta_cls: d x d
    xc = x - mu_cls
    Xt = xc.unsqueeze(-2)  # n x 1 x d
    X = xc.unsqueeze(-1)  # n x d x 1
    score = (Xt @ X / lda - Xt @ delta_cls @ X) / t  # n x 1 x 1
    dist = Xt @ X
    return score.flatten().clone(), dist.flatten().clone()


def score_full_cov(x, mu_cls, cov_cls, t=1):
    # x: n x d
    # mu_cls: d
    # delta_cls: d x d
    xc = x - mu_cls
    Xt = xc.unsqueeze(-2)  # n x 1 x d
    X = xc.unsqueeze(-1)  # n x d x 1
    # Sigma_cls_inv = torch.linalg.inv(cov_cls+0.01*torch.eye(cov_cls.shape[0]).cuda()) # avoid singular cov, but equivalent to add a probablistic term
    Sigma_cls_inv = torch.linalg.inv(0.01 * torch.eye(cov_cls.shape[0]).cuda())  # experiment on fixed covariance
    score = (Xt @ Sigma_cls_inv @ X) / t  # n x 1 x 1
    dist = Xt @ X
    return score.flatten().clone(), dist.flatten().clone()


def prediction(x_val, y_val, Nj, num_cls, mu, L, S, lda, t, device, num_batches=10):
    d = x_val.shape[1]
    N_val = x_val.shape[0]

    mu_online = mu.clone()
    L_online = L.clone()
    S_online = S.clone()

    y_pred = torch.empty(0, device=device)
    loss = torch.tensor(0.0, device=device)

    delta = torch.zeros((num_cls, d, d), device=device)
    for j in range(num_cls):
        delta[j] = deltaDiag(Nj[j], L_online[j], S_online[j], lda)

    size_batch = int(N_val / num_batches)
    valset = MixtureDataset(x_val, y_val)
    val_loader = DataLoader(valset, batch_size=size_batch, shuffle=False, drop_last=False)
    for id_batch, data in enumerate(val_loader):
        val_values, val_labels = data[0], list(data[1].numpy())
        x_batch = val_values.to(device)

        num_all_batch = x_batch.shape[0]
        # task: EM for PPCA - mix labeled and unlabeled together
        score_X = torch.zeros((num_cls, num_all_batch), device=device)
        dist_X = torch.zeros((num_cls, num_all_batch), device=device)
        for j in range(num_cls):
            score_X[j, :], dist_X[j, :] = score(x_batch, mu_online[j], delta[j], lda, t)

        y_pred_batch = torch.argmin(score_X, dim=0)
        y_pred_batch.int()
        for i in range(num_all_batch):
            loss += score_X[y_pred_batch[i], i]
        y_pred = torch.cat((y_pred, y_pred_batch), dim=0)

        del score_X, y_pred_batch, dist_X
        torch.cuda.empty_cache()
    return y_pred.clone(), loss.clone()


def prediction_train(train_loader, Nj, num_cls, mu, L, S, lda, t, device):
    d = mu.shape[1]

    mu_online = mu.clone()
    L_online = L.clone()
    S_online = S.clone()

    y_pred = torch.empty(0, device=device)
    loss = torch.tensor(0.0, device=device)

    delta = torch.zeros((num_cls, d, d))
    for j in range(num_cls):
        delta[j] = deltaDiag(Nj[j], L_online[j], S_online[j], lda).cpu()

    for id_batch, data in enumerate(train_loader):
        train_values, train_labels = data[0], list(data[1].numpy())
        x_batch = train_values.to(device)

        num_all_batch = x_batch.shape[0]
        # task: EM for PPCA - mix labeled and unlabeled together
        score_X = torch.zeros((num_cls, num_all_batch), device=device)
        dist_X = torch.zeros((num_cls, num_all_batch), device=device)
        for j in range(num_cls):
            delta_j = delta[j].to(device).clone()
            score_X[j, :], dist_X[j, :] = score(x_batch, mu_online[j], delta_j, lda, t)

        y_pred_batch = torch.argmin(score_X, dim=0)
        y_pred_batch.int()
        for i in range(num_all_batch):
            loss += score_X[y_pred_batch[i], i]
        y_pred = torch.cat((y_pred, y_pred_batch), dim=0)

        del score_X, y_pred_batch, dist_X
        torch.cuda.empty_cache()
    return y_pred.clone(), loss.clone()


def prediction_full_cov(x_val, y_val, num_cls, mu, full_cov, t, device, num_batches=10):
    N_val = x_val.shape[0]

    mu_online = mu.clone()
    y_pred = torch.empty(0, device=device)
    loss = torch.tensor(0.0, device=device)

    size_batch = int(N_val / num_batches)
    valset = MixtureDataset(x_val, y_val)
    val_loader = DataLoader(valset, batch_size=size_batch, shuffle=False, drop_last=False)
    for id_batch, data in enumerate(val_loader):
        val_values, val_labels = data[0], list(data[1].numpy())
        x_batch = val_values.to(device)

        num_all_batch = x_batch.shape[0]
        # task: EM for PPCA - mix labeled and unlabeled together
        score_X = torch.zeros((num_cls, num_all_batch), device=device)
        dist_X = torch.zeros((num_cls, num_all_batch), device=device)
        for j in range(num_cls):
            score_X[j, :], dist_X[j, :] = score_full_cov(x_batch, mu_online[j], full_cov[j], t)  # Full Sigma

        y_pred_batch = torch.argmin(score_X, dim=0)
        y_pred_batch.int()
        for i in range(num_all_batch):
            loss += score_X[y_pred_batch[i], i]
        y_pred = torch.cat((y_pred, y_pred_batch), dim=0)

        del score_X, y_pred_batch, dist_X
        torch.cuda.empty_cache()
    return y_pred.clone(), loss.clone()


def prediction_train_full_cov(train_loader, num_cls, mu, full_cov, t, device):
    mu_online = mu.clone()
    y_pred = torch.empty(0, device=device)
    loss = torch.tensor(0.0, device=device)

    for id_batch, data in enumerate(train_loader):
        train_values, train_labels = data[0], list(data[1].numpy())
        x_batch = train_values.to(device)

        num_all_batch = x_batch.shape[0]
        # task: EM for PPCA - mix labeled and unlabeled together
        score_X = torch.zeros((num_cls, num_all_batch), device=device)
        dist_X = torch.zeros((num_cls, num_all_batch), device=device)
        for j in range(num_cls):
            score_X[j, :], dist_X[j, :] = score_full_cov(x_batch, mu_online[j], full_cov[j], t)  # full Sigma

        y_pred_batch = torch.argmin(score_X, dim=0)
        y_pred_batch.int()
        for i in range(num_all_batch):
            loss += score_X[y_pred_batch[i], i]
        y_pred = torch.cat((y_pred, y_pred_batch), dim=0)

        del score_X, y_pred_batch, dist_X
        torch.cuda.empty_cache()
    return y_pred.clone(), loss.clone()


def EM_loss(log_ppi_prior):
    return -torch.sum(log_ppi_prior, dim=0)



class RAVE:
    def __init__(self):
        self.n = 0

    def add(self, X, y):
        n, p = X.shape
        Sx = torch.sum(X, dim=0)
        Sy = torch.sum(y)
        Sxx = X.t() @ X
        Sxy = X.t() @ y
        Syy = y.t() @ y
        if self.n == 0:
            self.n = n
            self.mx = Sx / n
            self.my = Sy / n
            self.mxx = Sxx / n
            self.mxy = Sxy / n
            self.myy = Syy / n
        else:
            self.mx = self.mx * (self.n / (self.n + n)) + Sx / (self.n + n)
            self.my = self.my * (self.n / (self.n + n)) + Sy / (self.n + n)
            self.mxx = self.mxx * (self.n / (self.n + n)) + Sxx / (self.n + n)
            self.mxy = self.mxy * (self.n / (self.n + n)) + Sxy / (self.n + n)
            self.myy = self.myy * (self.n / (self.n + n)) + Syy / (self.n + n)
            self.n = self.n + n

    def add_Weighted_onlyX(self, X, weights):
        # Gamma is a diagonal matrix with weights as the diagonal elements
        Gamma = torch.diag(weights)
        N = torch.sum(weights)
        Sx = X.t() @ weights  # p x 1
        Sxx = X.t() @ Gamma @ X  # p x p
        if self.n == 0:
            self.n = N
            self.mx = Sx / N
            self.mxx = Sxx / N
        else:
            self.mx = self.mx * (self.n / (self.n + N)) + Sx / (self.n + N)
            self.mxx = self.mxx * (self.n / (self.n + N)) + Sxx / (self.n + N)
            self.n = self.n + N

    def add_onlyX(self, X, mxx_cpu=False):
        n, p = X.shape
        Sx = torch.sum(X, dim=0)  # p x 1
        Sxx = (X.t() @ X)  # p x p
        if self.n == 0:
            self.n = n
            self.mx = Sx / n
            self.mxx = Sxx / n
        else:
            self.mx = (self.mx * (self.n / (self.n + n)) + Sx / (self.n + n))
            self.mxx = (self.mxx.cuda() * (self.n / (self.n + n)) + Sxx / (self.n + n))
            self.n = (self.n + n)
        if mxx_cpu:
            self.mxx = self.mxx.cpu()

    def add_score(self, X):
        n = len(X)
        Sx = torch.sum(X)  # 1
        Sxx = X.t() @ X  # 1
        if self.n == 0:
            self.n = n
            self.mx = Sx / n
            self.mxx = Sxx / n
        else:
            self.mx = self.mx * (self.n / (self.n + n)) + Sx / (self.n + n)
            self.mxx = self.mxx * (self.n / (self.n + n)) + Sxx / (self.n + n)
            self.n = self.n + n

    def add_rave(self, rave):
        n = rave.n
        if self.n == 0:
            self.n = rave.n
            self.mx = rave.mx.clone()
            self.my = rave.my.clone()
            self.mxx = rave.mxx.clone()
            self.mxy = rave.mxy.clone()
            self.myy = rave.myy.clone()
        else:
            n0 = self.n / (self.n + n)
            n1 = n / (self.n + n)
            self.mx = self.mx * n0 + rave.mx * n1
            self.my = self.my * n0 + rave.my * n1
            self.mxx = self.mxx * n0 + rave.mxx * n1
            self.mxy = self.mxy * n0 + rave.mxy * n1
            self.myy = self.myy * n0 + rave.myy * n1
            self.n = self.n + n

    def standardize_x(self):
        # standardize the raves for x
        var_x = torch.diag(self.mxx) - self.mx ** 2
        std_x = torch.sqrt(var_x)
        Pi = 1 / std_x

        XXn = self.mxx - self.mx.view(-1, 1) @ self.mx.view(1, -1)
        XXn *= Pi.view(1, -1)
        XXn *= Pi.view(-1, 1)

        return (XXn, Pi)

    def cov_weighted(self):
        self.mxx = self.mxx.cuda()
        self.mx = self.mx.cuda()
        XXn = self.mxx - self.mx.view(-1, 1) @ self.mx.view(1, -1)
        return XXn.clone()

    def cov_score(self):
        XXn = self.mxx - self.mx ** 2
        return XXn

    def standardize(self):
        # standardize the raves
        XXn, Pi = self.standardize_x()

        Temp1 = Pi * self.mxy
        Temp2 = self.my * Pi * self.mx
        XYn = Temp1 - Temp2

        return (XXn, XYn, Pi)


# utils
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    # import random
    # random.seed(seed)
    torch.backends.cudnn.deterministic = True


def kPPCA(num_labeled_per_class, q_ini, q_kPPCA, lda, num_iters, dataset_name, self_learner_name, feature_address,
          checkpoint_folder_address, run_id, sessions_list, seeds_list):
    num_sessions = len(sessions_list)
    for session_id in range(num_sessions):
        seed = seeds_list[session_id]
        setup_seed(seed)
        cls_range = sessions_list[session_id]
        num_prestored_cls = cls_range[0]
        num_new_cls = cls_range[1] - cls_range[0]
        print(
            'This is {}th run, session of {}th-{}th classes. {} classes in this session, {} classes has been processed'
            .format(run_id + 1, cls_range[0] + 1, cls_range[1], num_new_cls, num_prestored_cls))


        # task: data preparation
        if 'cifar100' in dataset_name.lower():
            train_set, val_set = CIFAR100_Feature(feature_address, cls_range=cls_range)  # for resnet x4
        if 'imagenet' in dataset_name.lower():
            train_set, val_set, folder_label_map = IMAGENET_Feature(feature_address,
                                                                    cls_range=cls_range)  # for resnet x4
        if 'cub' in dataset_name.lower():
            train_set,val_set, folder_label_map= CUB200_Feature(feature_address,cls_range=cls_range)

        # randomize training set
        train_loader = DataLoader(train_set, batch_size=len(train_set), shuffle=True)
        for it, data in enumerate(train_loader):
            train_values, train_labels, train_ori_id = data[0], list(data[1].numpy()), data[2]

        val_values, val_labels = val_set.values, val_set.labels  # for IMAGE_Feature()

        # save .mat data
        tr_name = 'SSFSCIL_{}_run{}_{}-{}cls_trainset_{}.pth'.format(
            dataset_name, run_id, cls_range[0] + 1, cls_range[1], self_learner_name)
        val_name = 'SSFSCIL_{}_run{}_{}-{}cls_valset_{}.pth'.format(
            dataset_name, run_id, cls_range[0] + 1, cls_range[1], self_learner_name)
        addr_tr = checkpoint_folder_address + tr_name
        addr_val = checkpoint_folder_address + val_name
        savemat(addr_tr,
                {'feature': train_values.cpu().float().numpy(),
                 'label': train_labels,
                 'original_id': train_ori_id.numpy()})
        savemat(addr_val,
                {'feature': val_values.cpu().float().numpy(),
                 'val_labels': val_labels})


        # task: Load prestored features
        # for normal size dimension mat
        tr_name = 'SSFSCIL_{}_run{}_{}-{}cls_trainset_{}.pth'.format(
            dataset_name, run_id, cls_range[0] + 1, cls_range[1], self_learner_name)
        val_name = 'SSFSCIL_{}_run{}_{}-{}cls_valset_{}.pth'.format(
            dataset_name, run_id, cls_range[0] + 1, cls_range[1], self_learner_name)
        addr_tr = checkpoint_folder_address + tr_name
        addr_val = checkpoint_folder_address + val_name

        train_data = loadmat(addr_tr)
        val_data = loadmat(addr_val)

        norm_name = 'SSFSCIL_{}_run{}_BaseSession_Normalization_{}.pth'.format(
            dataset_name, run_id, self_learner_name)
        norm_base_session_address = checkpoint_folder_address + norm_name
        if session_id == 0:
            # normalization for train and val data
            m_train_values = np.mean(train_data['feature'], axis=0)
            std_train_values = np.std(train_data['feature'], axis=0)
            d = m_train_values.shape[0]
            train_data['feature'] = (train_data['feature'] - m_train_values) / std_train_values / np.sqrt(d)
            m_val_values = np.mean(val_data['feature'], axis=0)
            std_val_values = np.std(val_data['feature'], axis=0)
            val_data['feature'] = (val_data['feature'] - m_val_values) / std_val_values / np.sqrt(d)

            torch.save({'m_train_values': m_train_values,
                        'std_train_values': std_train_values,
                        'm_val_values': m_val_values,
                        'std_val_values': std_val_values
                        },
                       norm_base_session_address)
        else:
            norm_basesession_result = torch.load(norm_base_session_address)
            m_val_basesession = norm_basesession_result['m_val_values']
            m_train_basesession = norm_basesession_result['m_train_values']
            std_val_basesession = norm_basesession_result['std_val_values']
            std_train_basesession = norm_basesession_result['std_train_values']
            d = m_train_basesession.shape[0]
            train_data['feature'] = (train_data['feature'] - m_train_basesession) / std_train_basesession / np.sqrt(d)
            val_data['feature'] = (val_data['feature'] - m_val_basesession) / std_val_basesession / np.sqrt(d)

        trainset = MixtureDataset(torch.tensor(train_data['feature'], dtype=torch.float), list(train_data['label'][0]))
        val_values, val_labels = torch.tensor(val_data['feature'], dtype=torch.float), list(val_data['val_labels'][0])
        N = len(trainset)


        # labeled by fixed number in class
        if session_id == 0:
            num_labeled = N
        else:
            num_labeled = num_new_cls * num_labeled_per_class


        # hyper-parameters
        t = 1  # holder for temperature annealling

        # obtain indicies for labeled data
        train_values, train_labels = trainset.values, trainset.labels
        num_batches = 80
        size_batch = math.ceil(len(trainset) / num_batches)

        # obtain indicies for labeled data - labeled by fixed number in class
        labeled_indices = []
        selectedobs_dict = {}
        if session_id == 0:  # base session, use all id as labeled (Tao's fscil setting)
            for i, y in enumerate(train_labels):
                if y not in selectedobs_dict.keys():
                    selectedobs_dict[y] = [i]
                    labeled_indices.append(i)
                else:
                    selectedobs_dict[y].append(i)
                    labeled_indices.append(i)
        else:
            for i, y in enumerate(train_labels):
                if y not in selectedobs_dict.keys():
                    selectedobs_dict[y] = [i]
                    labeled_indices.append(i)
                else:
                    if len(selectedobs_dict[y]) < num_labeled_per_class:
                        selectedobs_dict[y].append(i)
                        labeled_indices.append(i)
                    else:
                        continue
                if len(labeled_indices) == num_labeled:
                    break

        # print labeled result
        labeled_indices = labeled_indices[:num_labeled]
        print('{} labeled per class, {} labeled samples in current session'
              .format(num_labeled_per_class, len(labeled_indices)))

        # select labeled data for initialization
        xl = train_values[labeled_indices, :]
        yl = [train_labels[i] for i in labeled_indices]
        num_cls = len(set(train_labels)) + num_prestored_cls
        print(xl.shape, len(yl))



        # task: load augmented labeled features
        addr_aug = 'G:/Research/k-PPCAs/NewExamples/data_augx10_miniImgNet_train_ViT-L14_224_768_ImageNetNorm/data_augx10_miniImgNet_train_CLIP_ViT-L14_224_768_ImageNetNorm.mat' # dir storing the augmented training data

        ##################################################################
        # run at the first time, generate an integrated dataset (only for CUB200, miniImageNet and ImageNet-1k)
        ##################################################################
        # ## - integrate feature-extractor-generated features into a single feature .mat file
        # Aug_alltr_feature_address = [os.getcwd() + '\\..\\..\\NewExamples\\data_augx10_miniImgNet_train_ViT-L14_224_768_ImageNetNorm\\']
        # # train_aug_set = combine_augmented_features(Aug_alltr_feature_address, n_features=640) # for resnet x4
        # # train_aug_set = combine_augmented_features(Aug_alltr_feature_address, n_features=2048) # for resnet50
        # # train_aug_set = combine_augmented_features(Aug_alltr_feature_address, n_features=512) # for ViTB32
        # train_aug_set = combine_augmented_features(Aug_alltr_feature_address, n_features=768) # for ViTL14
        # train_aug_values=train_aug_set.values
        # train_aug_labels=train_aug_set.labels # list
        # savemat(addr_aug,{'feature': train_aug_values.cpu().float().numpy(), 'label': train_aug_labels})
        ####################################################################

        train_aug_data = loadmat(addr_aug)

        labeled_original_ids = train_ori_id[labeled_indices]
        aug_per_img = 10
        id_aug = []
        for id_loop, l_ori_id in enumerate(labeled_original_ids):
            start_aug_id = l_ori_id * aug_per_img
            end_aug_id = (l_ori_id + 1) * aug_per_img
            id_aug1 = [i1 for i1 in range(start_aug_id, end_aug_id)]
            id_aug.extend(id_aug1)
        xl_aug = torch.tensor(train_aug_data['feature'], dtype=torch.float)[id_aug]
        yl_aug = list(np.array(train_aug_data['label'][0])[id_aug])


        # normalization for augmented and labeled train data
        if session_id == 0:
            ## normalization for train and val data
            xl_aug = (xl_aug - m_train_values) / std_train_values / np.sqrt(d)
            # xl_aug = torch.tensor(xl_aug, dtype=torch.float)
        else:
            xl_aug = (xl_aug - m_train_basesession) / std_train_basesession / np.sqrt(d)
            # xl_aug = torch.tensor(xl_aug, dtype=torch.float)

        # concatenate augmented labeled data to labeled data
        # update trainset, xl, yl, labled_indices, N, num_batches
        trainset.values = torch.cat((trainset.values, xl_aug), dim=0)
        trainset.labels.extend(yl_aug)
        trainset.len = len(trainset.labels)
        trainset.original_id.extend([-1 for i in range(N, trainset.len)])

        xl = torch.cat((xl, xl_aug), dim=0)
        yl.extend(yl_aug)
        labeled_indices.extend([i for i in range(N, trainset.len)])
        N = len(trainset)

        num_batches = math.ceil(len(trainset) / size_batch)
        print('there are {} batches'.format(num_batches))

        xl = xl.to(device)

        # task: initial PCAs
        pi_ini = torch.tensor([1 / num_cls], device=device).repeat(num_cls).unsqueeze(-1)
        mu_ini = torch.zeros((num_cls, d), device=device)
        for cls, label in enumerate(sorted(list(set(train_labels)))):
            mu_ini[cls] = torch.mean(xl[torch.tensor(yl) == label, :], dim=0)

        L_ini = torch.zeros((num_cls, q_kPPCA, d), device=device)
        S_ini = torch.zeros((num_cls, q_kPPCA), device=device)
        Nj_ini = torch.zeros((num_cls, 1), device=device)
        for i in range(num_prestored_cls, num_cls):
            id_labeled = torch.tensor(yl) == i
            Nj_ini[i] = torch.sum(id_labeled)
            u, s, v = torch.linalg.svd(xl[id_labeled, :] - mu_ini[i, :])
            L_ini[i, 0:q_ini, :] = v[0:q_ini, :]
            s_fixedlenth = torch.zeros(q_ini)
            minl = min(q_ini, len(s))
            s_fixedlenth[:minl] = s[:minl]
            S_ini[i, 0:q_ini] = s_fixedlenth
            if i % 100 == 0:
                print(i)

        del id_labeled

        k = q_kPPCA  # q in kPPCA

        # recall previous stored classes and save to mu_ini, L_ini, S_ini, and Nj
        if session_id > 0:
            checkpoint_iter_old_session_name = 'SSFSCIL_{}_run{}_first{}cls_{}_PCAk{}_{}shot_k{}lda{:.0e}_iter{}.pth'.format(
                dataset_name, run_id, cls_range[0], self_learner_name, q_ini, num_labeled_per_class, k, lda, num_iters)
            prestored = checkpoint_folder_address + checkpoint_iter_old_session_name
            # {'mu_online': mu_online,
            #  'L_online': L_online,
            #  'Sx_online': S_online,
            #  'Scov_online': s_online,
            #  'lda': lda,
            #  't': t,
            #  'N': N,
            #  'Nj': Nj, 'num_cls': num_cls,
            #  'trainset': trainset,
            #  'val_values': val_values,
            #  'val_labels': val_labels
            #  }
            prestored_result = torch.load(prestored)

            Nj_prestored = prestored_result['Nj']
            mu_prestored = prestored_result['mu_online']
            L_prestored = prestored_result['L_online']
            S_prestored = prestored_result['Sx_online']

            Nj_ini[:num_prestored_cls] = Nj_prestored
            mu_ini[:num_prestored_cls] = mu_prestored
            L_ini[:num_prestored_cls] = L_prestored
            S_ini[:num_prestored_cls] = S_prestored

        addr_ini = checkpoint_folder_address + 'SSFSCIL_{}_run{}_{}-{}cls_{}_PCAk{}_{}shot_k{}lda{:.0e}_ini.pth'.format(
            dataset_name, run_id, cls_range[0] + 1, cls_range[1], self_learner_name, q_ini, num_labeled_per_class, k,
            lda)
        torch.save({'pi_ini': pi_ini,
                    'mu_ini': mu_ini,
                    'L_ini': L_ini,
                    'S_ini': S_ini,
                    'Nj_ini': Nj_ini,
                    # 'full_cov_ini': cov_ini,
                    'labeled_indices': labeled_indices,
                    'trainset': trainset},
                   addr_ini)
        print('initialization done')


        addr_ini = checkpoint_folder_address + 'SSFSCIL_{}_run{}_{}-{}cls_{}_PCAk{}_{}shot_k{}lda{:.0e}_ini.pth'.format(
            dataset_name, run_id, cls_range[0] + 1, cls_range[1], self_learner_name, q_ini, num_labeled_per_class, k,
            lda)

        ini_result = torch.load(addr_ini)
        mu_ini = ini_result['mu_ini']
        L_ini = ini_result['L_ini']
        S_ini = ini_result['S_ini']
        Nj_ini = ini_result['Nj_ini']

        torch.cuda.empty_cache()

        # task: generate indecies of labeled data in each batch
        labeled_indices_local_batches = [[] for i in range(num_batches)]
        for obs1_global_index in labeled_indices:
            obs1_local_index = obs1_global_index % size_batch
            true_batch_id = math.floor(obs1_global_index / size_batch)
            labeled_indices_local_batches[true_batch_id].append(obs1_local_index)

        # task: dataloader
        train_loader = DataLoader(trainset, batch_size=size_batch, shuffle=False, drop_last=False)


        mu_online = mu_ini.clone()
        L_online = L_ini.clone()
        S_online = S_ini.clone()
        Nj = Nj_ini.clone()

        del pi_ini, mu_ini, L_ini, S_ini, ini_result
        torch.cuda.empty_cache()

        # task: initial accuracy
        y_pred, loss = prediction(val_values, val_labels, Nj, num_cls,
                                           mu_online, L_online, S_online,
                                           lda, t, device)

        # use full covariance matrix
        # y_pred, loss = prediction_full_cov(val_values, val_labels, num_cls,
        #                                             mu_online, cov_online,
        #                                             t, device)

        y_true = torch.tensor(val_labels, dtype=y_pred.dtype).to(device)
        cls_err = torch.sum(y_pred != y_true) / (len(val_labels))
        print('Initial Accuracy rate - val: {:.4f}'.format(1 - cls_err.item()))
        print('Initial Loss - val:', loss.item())


        # training
        torch.cuda.empty_cache()
        print('there are {} classes'.format(num_cls))
        for i in range(num_iters):
            raves = [RAVE() for j in range(num_cls)]

            logdet_si = torch.zeros((num_cls, 1), device=device)
            delta = torch.zeros((num_cls, d, d))  # store at cpu

            if i == 0:
                for j in range(num_cls):
                    logdet_si[j] = logdet_cov(Nj[j], S_online[j], d, lda)
                    delta[j] = deltaDiag(Nj[j], L_online[j], S_online[j], lda).cpu()
            else:
                for j in range(num_cls):
                    logdet_si[j] = logdet_cov(Nj[j], S_online[j], d, lda)
                    delta[j] = deltaDiag(Nj[j], L_online[j], S_online[j], lda).cpu()

            for id_batch, data in enumerate(train_loader):
                train_values, train_labels = data[0], list(data[1].numpy())
                x_batch = train_values.to(device)
                num_all_batch = x_batch.shape[0]

                labeled_indices_1batch = labeled_indices_local_batches[id_batch]
                num_labeled_batch = len(labeled_indices_1batch)

                # task: kMeans for PPCA - mix labeled and unlabeled together
                score_X = torch.zeros((num_cls, num_all_batch), device=device)
                for j in range(num_cls):
                    delta_j = delta[j].to(device)
                    score_X[j], dist = score(x_batch, mu_online[j], delta_j, lda, t)
                    # score_X[j], dist = score_full_cov(x_batch, mu_online[j], cov_online[j], t)  # Use Full Covariance matrix


                # generate filter id based on score
                score_X_filterid = torch.zeros((num_cls, num_all_batch), dtype=torch.bool, device=device)
                score_minval, score_label = torch.min(score_X, dim=0)  # num_all_batch x 1
                score_X_labelmask = score_X == score_minval

                # create labeled samples' mask for each class
                score_X_labeled = torch.zeros((num_cls, num_labeled_batch), dtype=torch.bool,
                                              device=device)  # num_cls x N
                if num_labeled_batch > 0:
                    for j in range(num_labeled_batch):
                        score_X_labeled[train_labels[labeled_indices_1batch[j]], j] = True

                for j in range(num_prestored_cls, num_cls):
                    if i == 0:  # no filter for iter 1
                        score_X_j_minmask = torch.ones(num_all_batch, dtype=torch.bool, device=device)
                    else:
                        score_mu_currentbat = torch.mean(score_X[j, score_X_labelmask[j]])
                        score_std_currentbat = torch.std(score_X[j, score_X_labelmask[j]])
                        # # filtering out-of-distribution obs
                        # score_X_j_minmask = score_X[j] < (score_mu_currentbat + 0 * score_std_currentbat)
                        # score_X_j_minmask = score_X[j] < (score_mu_currentbat + 1 * score_std_currentbat)
                        # score_X_j_minmask = score_X[j] < (score_mu_currentbat + 1.64 * score_std_currentbat)
                        score_X_j_minmask = score_X[j] < (score_mu_currentbat + 1.96 * score_std_currentbat)
                        # score_X_j_minmask = score_X[j] < (score_mu_currentbat + 2.56 * score_std_currentbat)
                        ## no filter
                        # score_X_j_minmask = torch.ones(num_all_batch, dtype=torch.bool, device=device)

                    score_X_filterid[j] = score_X_j_minmask & score_X_labelmask[j]

                    score_X_filterid[j, labeled_indices_1batch] = score_X_filterid[j, labeled_indices_1batch] | \
                                                                  score_X_labeled[j]

                    raves[j].add_onlyX(x_batch[score_X_filterid[j], :].clone(), mxx_cpu=True)
                del x_batch, score_X, dist
                del score_X_filterid, score_label, score_X_labeled, score_X_labelmask, score_minval
                torch.cuda.empty_cache()

            # move raves mxx mx to cpu()
            for j in range(num_prestored_cls, num_cls):
                raves[j].mxx = raves[j].mxx.cpu()
                raves[j].mx = raves[j].mx.cpu()

            del delta, logdet_si
            torch.cuda.empty_cache()

            if session_id == 0:
                Nj = torch.zeros((num_cls, 1), device=device)
                mu_online = torch.zeros((num_cls, d), device=device)
            else:
                Nj = Nj.clone()
                mu_online = mu_online.clone()

            cov_online = torch.zeros((num_cls, d, d))
            for j in range(num_prestored_cls, num_cls):
                Nj[j] = raves[j].n
                mu_online[j] = raves[j].mx.to(device).clone()
                cov_online[j] = raves[j].cov_weighted().clone().cpu()

            if i > 0:
                del score_X_j_minmask, score_mu_currentbat, score_std_currentbat

            mu_online = mu_online.cpu()

            del raves
            torch.cuda.empty_cache()

            if session_id == 0:
                L_online = torch.zeros((num_cls, k, d), device=device)
                S_online = torch.zeros((num_cls, k), device=device)
            else:
                L_online = L_online.clone()
                S_online = S_online.clone()
            for stp in range(num_prestored_cls, num_cls):
                cov_online_cls = cov_online[stp].to(device)
                vT, s_online, v_online = torch.linalg.svd(cov_online_cls)
                L_online[stp] = v_online[0:k, :].clone()
                S_online[stp] = torch.sqrt(s_online[0:k] * Nj[stp]).clone()
                if stp % 100 == 0:
                    print('svd step:{} cls'.format(stp))
            print('finish update in this iteration')
            mu_online = mu_online.to(device)

            if (i + 1) % 5 == 0:
                checkpoint_iter_name = 'SSFSCIL_{}_run{}_first{}cls_{}_PCAk{}_{}shot_k{}lda{:.0e}_iter{}.pth'.format(
                    dataset_name, run_id, cls_range[1], self_learner_name, q_ini, num_labeled_per_class, k, lda, i + 1)
                torch.save({'mu_online': mu_online,
                            'L_online': L_online,
                            'Sx_online': S_online,
                            'Scov_online': s_online,
                            'lda': lda,
                            't': t,
                            'N': N,
                            'Nj': Nj,
                            'num_cls': num_cls,
                            'trainset': trainset,
                            'val_values': val_values,
                            'val_labels': val_labels
                            },
                           checkpoint_folder_address + checkpoint_iter_name)

            del s_online, v_online, vT
            del cov_online
            torch.cuda.empty_cache()

            # validation set evaluation
            print(min(val_labels), max(val_labels))
            y_pred_val_on, loss = prediction(val_values, val_labels, Nj, num_cls, mu_online, L_online,
                                                      S_online,
                                                      lda, t,
                                                      device)
            y_true_val = torch.tensor(val_labels, dtype=y_pred_val_on.dtype).to(device)
            cls_err_val_on = torch.sum(y_pred_val_on != y_true_val) / (len(val_labels))
            acc_val_on = 1 - cls_err_val_on.item()
            print('#{} Accuracy rate - val: {:.4f}'.format(i, acc_val_on))
            print('#{} loss - on: {}'.format(i, loss.item()))

            if (i + 1) == num_iters:
                trend_name = 'SSFSCIL_{}_run{}_first{}cls_acc_val_trend.pth'.format(
                    dataset_name, run_id, cls_range[1])
                torch.save(
                    {'acc_val': acc_val_on},
                    checkpoint_folder_address + trend_name)

            del y_pred_val_on, y_true_val, cls_err_val_on, loss
            torch.cuda.empty_cache()
            print('End of one iteration')

            if (i + 1) == num_iters:
                # task: trainset evaluation
                y_pred, loss = prediction_train(train_loader, Nj, num_cls, mu_online,  # pi_online,
                                                         L_online, S_online, lda, t,
                                                         device)
                y_true = torch.tensor(trainset.labels, dtype=y_pred.dtype).to(device)
                cls_err = torch.sum(y_pred != y_true) / (len(trainset.labels))  # (len(val_labels))
                print('Accuracy rate - train: {:.4f}'.format(1 - cls_err.item()))
                print('Loss - train:', loss.item())


if __name__ == '__main__':
    # device, seed settings
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    # runs_seeds_list=[[380, 27, 214, 8, 489, 52, 66, 216, 225, 273, 79],
    #                  [170, 211, 390, 363, 21, 243, 424, 276, 479, 318, 446],
    #                  [371, 383, 466, 42, 478, 222, 74, 410, 404, 216, 12],
    #                  [134, 339, 402, 450, 232, 129, 142, 65, 137, 224, 153],
    #                  [147, 75, 37, 256, 238, 387, 111, 72, 111, 347, 435]]  # for CUB200
    # sessions_list=[[0,100],
    #                [100,110],
    #                [110,120],
    #                [120,130],
    #                [130,140],
    #                [140,150],
    #                [150,160],
    #                [160,170],
    #                [170,180],
    #                [180,190],
    #                [190,200]]  # for CUB200

    runs_seeds_list = [[5, 272, 400, 498, 441, 381, 368, 226, 16],
                       [210, 380, 24, 415, 61, 231, 440, 260, 417],
                       [239, 58, 427, 490, 8, 83, 251, 460, 327],
                       [140, 226, 269, 186, 206, 418, 131, 481, 64],
                       [381, 125, 345, 491, 298, 95, 430, 253, 369]]  # for miniImageNet
    sessions_list = [[0, 60], [60, 65], [65, 70], [70, 75], [75, 80], [80, 85], [85, 90], [90, 95], [95, 100]]  # for CIFAR100 and miniImageNet

    # runs_seeds_list=[[5,88,387,113,471,135],
    #                  [491,122,232,18,470,210],
    #                  [224,77,212,95,10,251],
    #                  [383,191,372,55,192,233],
    #                  [293,328,306,37,137,452]]
    # sessions_list=[[0,500],[500,600],[600,700],[700,800],[800,900],[900,1000]]  # for ImageNet-1k


    num_runs = len(runs_seeds_list)

    # feature_address = [os.getcwd() + '\\..\\benchmarks\\CUB200\\CUB200_224_ImgNetNorm_ViTL14\\']
    # feature_address = [os.getcwd() + '\\..\\benchmarks\\CIFAR100\\CIFAR100_224_ImgNetNorm_ViTL14\\']
    feature_address = [os.getcwd() + '\\..\\benchmarks\\Mini-ImageNet\\miniImgNet_224_ImgNetNorm_ViTL14\\']
    # feature_address = [os.getcwd() + '\\..\\benchmarks\\ImageNet_CLIP\\']

    checkpoint_folder_addr = os.getcwd() + '\\..\\checkpoint\\'

    dataset_name = 'augMiniImageNet_224x224'
    self_learner_name = 'CLIP_ViTL14'

    num_labeled_per_class = 5
    q_ini = 10
    q_kPPCA = 10
    lda = 1e-2
    num_iters = 10

    for run_id in range(num_runs):
        seeds_list = runs_seeds_list[run_id]
        kPPCA(num_labeled_per_class,
              q_ini,
              q_kPPCA,
              lda,
              num_iters,
              dataset_name,
              self_learner_name,
              feature_address,
              checkpoint_folder_addr,
              run_id,
              sessions_list,
              seeds_list)
