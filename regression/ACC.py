import sys
sys.path.append("/home/xinyi/maplecg_nfs/codebase/Data_Distribution_Valuation")

from copy import deepcopy
from utils import cwd, set_deterministic, save_results

from data_utils import _get_loader
from utils import get_trained_feature_extractor, get_accuracy
import numpy as np
import torch

from reg_data_utils import  _get_CaliH, _get_KingH, _get_FaceA, _get_census, huber_regression, assign_data

from tqdm import tqdm
import argparse

from os.path import join as oj

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

baseline = 'ground_truth'


if __name__ == '__main__':    
    print(f"----- Running experiment for {baseline} -----")

    parser = argparse.ArgumentParser(description='Process which dataset to run for regression.')
    parser.add_argument('-N', '--N', help='Number if data venrods.', type=int, required=True, default=5)
    parser.add_argument('-m', '--size', help='Size of sample datasets.', type=int, required=True, default=1500)
    parser.add_argument('-P', '--dataset', help='Pick the dataset to run.', type=str, required=True)
    parser.add_argument('-Q', '--Q_dataset', help='Pick the Q dataset.', type=str, required=True, choices=['KingH', 'Census17'])
    parser.add_argument('-n_t', '--n_trials', help='Number of trials.', type=int, default=5)

    # parser.add_argument('-nocuda', dest='cuda', help='Not to use cuda even if available.', action='store_false')
    # parser.add_argument('-cuda', dest='cuda', help='Use cuda if available.', action='store_true')

    cmd_args = parser.parse_args()
    print(cmd_args)

    dataset = cmd_args.dataset
    Q_dataset = cmd_args.Q_dataset
    N = cmd_args.N
    size = cmd_args.size
    n_trials = cmd_args.n_trials
    GT_size = 10000 # Ground truth size for distribution performance

    set_deterministic()
    
    if not (dataset == 'CaliH' or dataset == 'Census15'):
        raise NotImplementedError(f"P = {dataset} is not implemented.")

    cod_over_trials = []
    ridge_cod_over_trials = []
    mses_over_trials = []
    for _ in tqdm(range(n_trials), desc =f'A total of {n_trials} trials.'):
        D_Xs, D_Ys, V_X, V_Y = assign_data(N, max(GT_size, 2*size), dataset, Q_dataset)
        scores = []
        ridge_scores = []
        mses = []
        for D_X, D_Y in zip(D_Xs, D_Ys):

            lin_reg = LinearRegression().fit(D_X, D_Y)
            score = lin_reg.score(V_X, V_Y)
            scores.append(score) # coefficient of determination

            ridge_reg = Ridge().fit(D_X, D_Y)
            score = ridge_reg.score(V_X, V_Y)
            ridge_scores.append(score) # coefficient of determination

            y_pred = ridge_reg.predict(V_X)
            mse = mean_squared_error(y_pred, V_Y)
            mses.append(mse)

        print('scores:', scores)
        print('ridge scores:', ridge_scores)
        print('mses:', mses)
        cod_over_trials.append(scores)
        ridge_cod_over_trials.append(ridge_scores)
        mses_over_trials.append(mses)

    results = {'mses_over_trials': mses_over_trials, 'ridge_cod_over_trials': ridge_cod_over_trials, 
               'cod_over_trials': cod_over_trials, 'N':N, 'size':size, 'n_trials': n_trials}
    exp_name =oj('regression', f'{dataset}_vs_{Q_dataset}-N{N} m{size} n_trials{n_trials}')
    save_results(baseline=baseline, exp_name=exp_name, **results)