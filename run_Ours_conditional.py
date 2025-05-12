import torch
from utils import set_deterministic, save_results, get_trained_feature_extractor, CNN_Net_32, LogisticRegression, get_trained_feature_extractor

from data_utils import assign_data, _get_loader, _get_CIFAR10, _get_MNIST32
from copy import deepcopy

from tqdm import tqdm
import argparse
from os.path import join as oj


baseline = 'Ours_conditional'

def get_conditional_samples(V_X, V_Y, model, device):
    '''
    Empirically approximates the distribution of Y given X (i.e., label given the feature) of (D_X, D_Y) as follows:
        1. train a classifier on (D_X, D_Y)
        2. feedforward dataset (e.g., reference of validation dataset) in to the trained classifier and collect the outputs
    '''
    model = model.to(device)
    eval_loader = _get_loader(V_X.to(device), V_Y.to(device), batch_size=1024, shuffle=False, drop_last=False)
    model.eval()
    with torch.no_grad():
        samples = [model(x) for x, _ in eval_loader]

    # use softmax to cast each prediction into a probability vector 
    samples = torch.softmax(torch.cat(samples, dim=0), dim=1)
    return samples

from mmd import batched_rbf_mmd2


def _paired_MMD(Y_given_X_samples, ground_truth_samples, device=torch.device('cuda'), sigma_list = [1, 2, 5, 10], batch_size=100):
    '''
    Using paired MMD via the batching operation:
        Pairing each batch in Y_given_X_sample to each batch of ground_truth_samples, because they are the prediction and (one-hot) labels
    of the same batch of inputs.

        While the usual shuffled MMD operations also seems to work empirically, this paired MMD makes more sense here, by conditioning on 
    the same batch of features.

        In particular, a smaller batch size leads to a more "accurate" evaluation. If batch_size=1, it becomes similar to cross entropy in 
    the sense that the overall (batch averaged) MMD is evaluating the per-sample difference (i.e., mean difference) between the prediction 
    and label in each pair. Note that cross-entropy is evaluating a different per-sample difference according to the log probabilities.
    '''
    MMD2s = [batched_rbf_mmd2(Y_given_X_sample, ground_truth_samples, sigma_list=sigma_list, device=device, batch_size=batch_size) for Y_given_X_sample in Y_given_X_samples]
    
    MMD2_values = [-MMD2.item() for MMD2 in MMD2s]

    min_MMD2 = min(MMD2s)
    if min_MMD2 < 0:
        # linearly translate up by the smallest so that the sqrt is well-defined
        MMD2s = [MMD2 + (- min_MMD2) for MMD2 in MMD2s]
    # take the suare root
    MMD_values = [-torch.sqrt(MMD2).item() for MMD2 in MMD2s]

    return MMD2_values, MMD_values

class options:
    epochs = 30
    mmd_batch_size = 1024

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Process which dataset to run')
    parser.add_argument('-N', '--N', help='Number of data vendors.', type=int, required=True, default=5)
    parser.add_argument('-m', '--size', help='Size of sample datasets.', type=int, required=True, default=1500)
    parser.add_argument('-P', '--dataset', help='Pick the dataset to run.', type=str, required=True)
    parser.add_argument('-Q', '--Q_dataset', help='Pick the Q dataset.', type=str, required=False, choices=['normal', 'EMNIST', 'FaMNIST', 'CIFAR100' , 'CreditCard', 'UGR16'])
    parser.add_argument('-n_t', '--n_trials', help='Number of trials.', type=int, default=5)
    parser.add_argument('-nh', '--not_huber', help='Not with huber, meaning with other types of specified heterogeneity.', action='store_true')
    parser.add_argument('-het', '--heterogeneity', help='Type of heterogeneity.', type=str, default='normal', choices=['normal', 'label', 'classimbalance', 'classimbalance_inter'])
    parser.add_argument('-kde', dest='gmm', help='Whether to use KDE for generator distribution. Only applicable to CreditCard or TON dataset.', action='store_false')
    parser.add_argument('-gmm', dest='gmm', help='Whether to use GMM for generator distribution. Only applicable to CreditCard or TON dataset.', action='store_true')
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
    use_GMM = cmd_args.gmm

    if dataset == 'MNIST':
        X_train, y_train, X_test, y_test = _get_MNIST32()
        n_classes = 10
        # MNIST
        model_ini = CNN_Net_32()        
        options.mmd_batch_size = 256

        options.epochs = 5

    elif dataset == 'CIFAR10':
        X_train, y_train, X_test, y_test = _get_CIFAR10()
        # CIFAR10
        import torchvision.models as models
        from torch import nn
        model_ini = models.resnet18(pretrained=True)
        model_ini.fc = nn.Linear(512, 10)
        n_classes = 10

        options.mmd_batch_size = 256
        options.epochs = 20

    elif dataset == 'CreditCard':
        model_ini = LogisticRegression(7, 2)
        n_classes = 2

    elif dataset == 'TON':
        model_ini = LogisticRegression(22, 8)
        n_classes = 8

    else:
        raise NotImplementedError(f"P = {dataset} is not implemented.")

    print(f"----- Running experiment for {baseline} -----")

    set_deterministic()
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    values_over_trials, values_hat_over_trials = [], []

    # the baseline of MMD2: 
    # when a reference is given, use the reference, but the valuation is based on MMD squared
    values_mmd2_over_trials, values_hat_mmd2_over_trials = [], []
    
    n_avgs = 5
    for _ in tqdm(range(n_trials), desc =f'A total of {n_trials} trials.'):
        # raw data
        D_Xs, D_Ys, V_X, V_Y, labels = assign_data(N, size, dataset, Q_dataset, not_huber, heterogeneity)

        reference = torch.cat(D_Xs) # this is P_N
        netD = None
        ground_truth_samples = torch.nn.functional.one_hot(V_Y, n_classes).float() # the one-hot encoded labels
        Y_given_X_samples = []
        Y_given_X_samples_hat = []
        try:
            models = []
            for D_X, D_Y in zip(D_Xs, D_Ys):
                train_loader = _get_loader(D_X, D_Y)
                trained_model = get_trained_feature_extractor(deepcopy(model_ini), train_loader, test_loader=None, device=device, epochs=options.epochs)
    
                samples = get_conditional_samples(V_X, V_Y, trained_model, device)
                Y_given_X_samples.append(samples)
                models.append(trained_model)

            # Note that the reference here is the one-hot encoded labels, NOT the features
            MMD2_values, MMD_values = _paired_MMD(Y_given_X_samples, ground_truth_samples)
            values_over_trials.append(MMD_values)

            values_mmd2_over_trials.append(MMD2_values)


            # for hat: no given validation set
            D_N_X, D_N_Y = torch.cat(D_Xs), torch.cat(D_Ys)
            ref_model = get_trained_feature_extractor(deepcopy(model_ini), train_loader=_get_loader(D_N_X, D_N_Y), test_loader=None, device=device, epochs=options.epochs)
            ref_samples = get_conditional_samples(D_N_X, D_N_Y, ref_model, device)

            for i, model in enumerate(models): # vendor i
                samples = get_conditional_samples(D_N_X, D_N_X, model, device)
                Y_given_X_samples_hat.append(samples)

            # Note that the reference here is the one-hot encoded labels, NOT the features
            MMD2_values, MMD_values = _paired_MMD(Y_given_X_samples_hat, ref_samples)
            values_hat_over_trials.append(MMD_values)

            values_hat_mmd2_over_trials.append(MMD2_values)


        except RuntimeError as e: # Cuda Memory issue
            if str(e).startswith('CUDA out of memory.'): print('CUDA out of memory.')            
            raise Exception

    if not_huber:
        exp_name = oj('not_huber', f'{dataset}_vs_{heterogeneity}-N{N} m{size} n_trials{n_trials}')
    else:
        exp_name = f'{dataset}_vs_{Q_dataset}-N{N} m{size} n_trials{n_trials}'

    results = {'values_over_trials': values_over_trials, 'values_hat_over_trials': values_hat_over_trials, 
    'N':N, 'size':size, 'n_trials': n_trials, 'isHuber':not not_huber, 'heterogeneity': heterogeneity, 'use_GMM': use_GMM}
    save_results(baseline=baseline, exp_name=exp_name, **results)

    results = {'values_over_trials': values_mmd2_over_trials, 'values_hat_over_trials': values_hat_mmd2_over_trials, 
    'N':N, 'size':size, 'n_trials': n_trials, 'isHuber':not not_huber, 'heterogeneity': heterogeneity, 'use_GMM': use_GMM}
    save_results(baseline='MMD_sq_half_mix_conditional', exp_name=exp_name, **results)
