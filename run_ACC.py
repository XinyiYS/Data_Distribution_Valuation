from copy import deepcopy

import torch
import numpy as np
from tqdm import tqdm
import argparse

from utils import cwd, set_deterministic, save_results, CNN_Net_32, LogisticRegression, get_trained_feature_extractor, get_accuracy
from data_utils import huber, assign_data, _get_loader

import torchvision.models as models
import torch.nn as nn
from os.path import join as oj


baseline = 'ground_truth'


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Process which dataset to run')
    parser.add_argument('-N', '--N', help='Number of data vendors.', type=int, required=True, default=5)
    parser.add_argument('-m', '--size', help='Size of sample datasets.', type=int, required=True, default=1500)
    parser.add_argument('-P', '--dataset', help='Pick the dataset to run.', type=str, required=True)
    parser.add_argument('-Q', '--Q_dataset', help='Pick the Q dataset.', type=str, required=False, choices=['normal', 'EMNIST', 'FaMNIST', 'CIFAR100' , 'CreditCard', 'UGR16'])
    parser.add_argument('-n_t', '--n_trials', help='Number of trials.', type=int, default=5)
    parser.add_argument('-nh', '--not_huber', help='Not with huber, meaning with other types of specified heterogeneity.', action='store_true')
    parser.add_argument('-het', '--heterogeneity', help='Type of heterogeneity.', type=str, default='normal', choices=['normal', 'label', 'classimbalance', 'classimbalance_inter'])

    # parser.add_argument('-nocuda', dest='cuda', help='Not to use cuda even if available.', action='store_false')
    # parser.add_argument('-cuda', dest='cuda', help='Use cuda if available.', action='store_true')

    cmd_args = parser.parse_args()
    print(cmd_args)


    dataset = cmd_args.dataset
    Q_dataset = cmd_args.Q_dataset
    N = cmd_args.N
    size = cmd_args.size
    n_trials = cmd_args.n_trials
    not_huber = cmd_args.not_huber
    heterogeneity = cmd_args.heterogeneity

    print(f"----- Running experiment for {baseline} -----")

    set_deterministic()
    GT_size = 10000 # size of dataset from P to represent the actual distribution P

    if dataset == 'MNIST':
        epochs = 60
        model_ini = CNN_Net_32()
    elif dataset == 'CIFAR10':
        epochs = 100
        model_ini = models.resnet18(pretrained=True)
        model_ini.fc = nn.Linear(512, 10)
    elif dataset == 'CreditCard':
        epochs = 30
        model_ini = LogisticRegression(7, 2)
    elif dataset == 'TON':
        epochs = 30
        model_ini = LogisticRegression(22, 8)
    else:
        raise NotImplementedError(f"P = {dataset} is not implemented.")

    accs_over_trials = []
    for _ in tqdm(range(n_trials), desc =f'A total of {n_trials} trials.'):
        # multiply a factor to the size to simulate the distribution of (D_X, D_Y)
        D_Xs, D_Ys, V_X, V_Y, labels = assign_data(N, max(GT_size, 2*size), dataset, Q_dataset, not_huber, heterogeneity)
        model = deepcopy(model_ini)

        accs = []
        test_loader = _get_loader(V_X, V_Y)
        for D_X, D_Y in zip(D_Xs, D_Ys):
            train_loader = _get_loader(D_X, D_Y, shuffle=True)
            model = get_trained_feature_extractor(model, train_loader, epochs=epochs)
            test_acc = get_accuracy(model, test_loader).item()
            accs.append(test_acc)

        print(f"Test accuracies are: {accs}.")
        accs_over_trials.append(accs)

    results = {'accs_over_trials': accs_over_trials, 'N':N, 'size':size, 
    'n_trials': n_trials, 'GT_size': GT_size, 'isHuber':not not_huber, 'heterogeneity': heterogeneity}
    if not_huber:
        exp_name =oj('not_huber', f'{dataset}_vs_{heterogeneity}-N{N} m{size} n_trials{n_trials}')
    else:
        exp_name =f'{dataset}_vs_{Q_dataset}-N{N} m{size} n_trials{n_trials}'
    save_results(baseline=baseline, exp_name=exp_name, **results)