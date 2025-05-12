import sys
# sys.path.append("/home/fengzi/Downloads/codebase/DataDistribution-Valuation/")
sys.path.append("/home/xinyi/maplecg_nfs/codebase/Data_Distribution_Valuation")

import torch
from copy import deepcopy
from utils import cwd, set_deterministic, save_results
from regression._Ours_utils import get_trained_regressor
from run_Ours_conditional import _paired_MMD

from data_utils import _get_loader
import numpy as np
import torch

from reg_data_utils import assign_data
from os.path import join as oj

from tqdm import tqdm
import argparse
from utils import save_results, MLPRegression, LinearRegression

from mmd import rbf_mmd2
baseline = 'Ours_conditional'

def get_conditional_regression_samples(V_X, V_Y, model, device):

    model = model.to(device)
    eval_loader = _get_loader(V_X.to(device), V_Y.to(device), batch_size=1024, shuffle=False, drop_last=False, mode='reg')
    model.eval()
    with torch.no_grad():
        samples = torch.cat([model(x) for x, _ in eval_loader], dim=0)

    return samples


def get_MMD_values_reg(Y_given_X_samples, ground_truth_samples, device=torch.device('cuda')):
    # for regression, the dimension is lower, so there is no need for the batch version to avoid CUDA OOM
    # use one-shot MMD computation directly
        
    MMD2s = [rbf_mmd2(Y_given_X, ground_truth_samples).item() for Y_given_X in Y_given_X_samples]
    return [- MMD2 for MMD2 in MMD2s], [- np.sqrt(max(1e-6, MMD2)) for MMD2 in MMD2s]

from scipy.stats import pearsonr
class options:
    cuda = True
    batch_size = 256
    epochs = 20
    learning_epochs = 20

if __name__ == '__main__':
    
    print(f"----- Running experiment for {baseline} -----")

    parser = argparse.ArgumentParser(description='Process which dataset to run for regression.')
    parser.add_argument('-N', '--N', help='Number if data venrods.', type=int, required=True, default=5)
    parser.add_argument('-m', '--size', help='Size of sample datasets.', type=int, required=True, default=1500)
    parser.add_argument('-P', '--dataset', help='Pick the dataset to run.', type=str, required=True)
    parser.add_argument('-Q', '--Q_dataset', help='Pick the Q dataset.', type=str, required=True, choices=['KingH', 'Census17'])
    parser.add_argument('-n_t', '--n_trials', help='Number of trials.', type=int, default=5)
    parser.add_argument('-gmm', dest='gmm', help='Whether to use GMM for generator distribution.', action='store_true')
    parser.add_argument('-kde', dest='gmm', help='Whether to use KDE for generator distribution.', action='store_false')

    parser.add_argument('-nocuda', dest='cuda', help='Not to use cuda even if available.', action='store_false')
    parser.add_argument('-cuda', dest='cuda', help='Use cuda if available.', action='store_true')

    cmd_args = parser.parse_args()
    print(cmd_args)

    dataset = cmd_args.dataset
    Q_dataset = cmd_args.Q_dataset
    N = cmd_args.N
    size = cmd_args.size
    n_trials = cmd_args.n_trials
    use_GMM = cmd_args.gmm
    cuda = cmd_args.cuda

    if torch.cuda.is_available() and cuda:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    set_deterministic()

    if dataset == 'CaliH':
        data_dim = 10
        model_ini = LinearRegression(data_dim)
        dims = tuple([data_dim])

    elif dataset == 'Census15':
        data_dim = 16
        model_ini = LinearRegression(data_dim)
        dims = tuple([data_dim])

    else:
        raise NotImplementedError(f"P = {dataset} is not implemented.")

    values_over_trials, values_hat_over_trials = [], []
    values_hat_no_gen_over_trials = []
    values_mmd2_over_trials, values_hat_mmd2_over_trials = [], []
    for _ in tqdm(range(n_trials), desc =f'A total of {n_trials} trials.'):
        # raw data
        D_Xs, D_Ys, V_X, V_Y = assign_data(N, size, dataset, Q_dataset)

        ground_truth_samples = V_Y
        Y_given_X_samples, Y_given_X_samples_hat = [], []
        models = []
        for D_X, D_Y in zip(D_Xs, D_Ys):

            train_loader = _get_loader(D_X, D_Y, drop_last=False, mode='reg')
            trained_model = get_trained_regressor(deepcopy(model_ini), train_loader, epochs=options.learning_epochs)

            samples = get_conditional_regression_samples(V_X, V_Y, trained_model, device)
            Y_given_X_samples.append(samples)
            models.append(trained_model)

        if torch.cuda.device_count() > 1:
            MMD2_values, MMD_values = get_MMD_values_reg(Y_given_X_samples, ground_truth_samples, device = torch.device("cuda:1"))
        else:
            MMD2_values, MMD_values = _paired_MMD(Y_given_X_samples, ground_truth_samples, batch_size=1024)

        values_over_trials.append(MMD_values)
        print(f"MMD values cond. pearsonr: {pearsonr(MMD_values, -np.arange(N)) }")

        values_mmd2_over_trials.append(MMD2_values)
        print(f"MMD2 values cond. pearsonr: {pearsonr(MMD2_values, -np.arange(N)) }")

        # release memory
        del Y_given_X_samples, ground_truth_samples

        # for hat: no given validation set
        D_N_X, D_N_Y = torch.cat(D_Xs), torch.cat(D_Ys)
        ref_model = get_trained_regressor(deepcopy(model_ini), train_loader=_get_loader(D_N_X, D_N_Y, drop_last=False,  mode='reg'), test_loader=None, device=device, epochs=options.epochs)
        ref_samples = get_conditional_regression_samples(D_N_X, D_N_Y, ref_model, device)

        for i, model in enumerate(models): # vendor i
            samples = get_conditional_regression_samples(D_N_X, D_N_X, model, device)
            Y_given_X_samples_hat.append(samples)

        if torch.cuda.device_count() > 1:
            MMD2_values, MMD_values = get_MMD_values_reg(Y_given_X_samples_hat, ref_samples, device = torch.device("cuda:1"))
        else:
            MMD2_values, MMD_values = _paired_MMD(Y_given_X_samples_hat, ref_samples, batch_size=1024)

        values_hat_over_trials.append(MMD_values)
        print(f"MMD values hat cond. pearsonr: {pearsonr(MMD_values, -np.arange(N)) }")

        values_hat_mmd2_over_trials.append(MMD2_values)
        print(f"MMD2 values hat cond. pearsonr: {pearsonr(MMD2_values, -np.arange(N)) }")


    # Ours cond.
    results = {'values_over_trials': values_over_trials, 'values_hat_over_trials': values_hat_over_trials, 
               'N':N, 'size':size, 'n_trials': n_trials, 'use_GMM': use_GMM}
    save_results(baseline=baseline, exp_name=oj('regression', f'{dataset}_vs_{Q_dataset}-N{N} m{size} n_trials{n_trials}'), **results)

    # For MMD squared w.r.t. half mix reference, cond.
    results = {'values_over_trials': values_mmd2_over_trials, 'values_hat_over_trials': values_hat_mmd2_over_trials,
            'N':N, 'size':size, 'n_trials': n_trials, 'use_GMM': use_GMM}
    # save_results(baseline='MMD_sq_half_mix', exp_name=exp_name, **results)
    save_results(baseline='MMD_sq_half_mix_conditional', exp_name=oj('regression', f'{dataset}_vs_{Q_dataset}-N{N} m{size} n_trials{n_trials}'), **results)
