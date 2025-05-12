
from tqdm import tqdm
import argparse
from os.path import join as oj

import torch
import torchvision.models as models
import torch.nn as nn

from mmd import rbf_mmd2, get_MMD_values_uneven
from utils import cwd, set_deterministic, save_results, CNN_Net_32, LogisticRegression, get_trained_feature_extractor, get_accuracy
from data_utils import huber, assign_data

from data_utils import _get_loader, _get_MNIST32, _get_CIFAR10, _get_credit_card, _get_TON


def get_MMD_values(D_Xs, D_Ys, V_X, V_Y, sigma_list =[1,2,5,10]):
    return [ -torch.sqrt(max(rbf_mmd2(D_X, V_X, sigma_list), 1e-6)) for D_X in D_Xs]

def get_extracted(model, loader, device):
    model = model.to(device)
    D_X = []
    model.eval()
    with torch.no_grad():
        for i, (batch_data, batch_target) in enumerate(loader):
            batch_data, batch_target = batch_data.to(device), batch_target.to(device)
            outputs = model(batch_data)

            D_X.append(outputs)

    return torch.cat(D_X)

# from sklearn.utils import resample
# def get_mix_reference(D_Xs, generated_reference, pct, device=torch.device('cuda')):
#     D_N = torch.cat(D_Xs).to(device)
#     generated_reference = generated_reference.to(device)
#     m = min(len(D_N), len(generated_reference))
#     D_N_sub = resample(D_N, n_samples=int((1 - pct)*m))
#     generated_reference_sub = resample(generated_reference, n_samples=int(pct * m))

#     print(f"intersections: {sum([ (_ in D_N) for _ in D_N_sub])} / {len(D_N)},  {sum([ (_ in generated_reference) for _ in generated_reference_sub])} / {len(generated_reference)}")
#     reference = torch.cat([D_N_sub, generated_reference_sub])
#     print(f"For {pct} of generated, {D_N_sub.shape}, {generated_reference_sub.shape}, and the mixed reference shape is : {reference.shape}")    
#     return reference


baseline = 'Ours'

'''
If a validatin set is available:
    the reference is the valiation set.
else:   
    the reference is the union of all datasets.

Using MMD instead of MMD2.

'''

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
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    if dataset == 'MNIST':
        X_train, y_train, X_test, y_test = _get_MNIST32()
        # MNIST
        model = CNN_Net_32()
        # latent dimension d
        d = 10
        epochs = 10

    elif dataset == 'CIFAR10':
        X_train, y_train, X_test, y_test = _get_CIFAR10()
        # CIFAR10
        model = models.resnet18(pretrained=True)
        model.fc = nn.Linear(512, 10)
        d = 10
        epochs = 50
    elif dataset == 'CreditCard':
        X_train, y_train, X_test, y_test = _get_credit_card()
        epochs = 30
        model = LogisticRegression(7, 2)
        d = 7
    elif dataset == 'TON':
        X_train, y_train, X_test, y_test = _get_TON()
        epochs = 30
        model = LogisticRegression(22, 8)
        d = 22
    else:
        raise NotImplementedError(f"P = {dataset} is not implemented.")

    # trainloader, testloader = _get_loader(X_train, y_train), _get_loader(X_test, y_test)
    # feature_extractor = get_trained_feature_extractor(model, trainloader, testloader, epochs=epochs)
    values_over_trials, values_hat_over_trials =[], []

    for _ in tqdm(range(n_trials), desc =f'A total of {n_trials} trials.'):
        # raw data
        D_Xs, D_Ys, V_X, V_Y, labels = assign_data(N, size, dataset, Q_dataset, not_huber, heterogeneity)

        # extract features for MMD
        # for i, (D_X, D_Y) in enumerate(zip(D_Xs, D_Ys)):
            # loader = _get_loader(D_X, D_Y)
            # D_Xs[i] = get_extracted(feature_extractor, loader, device)

        # VX_loader = _get_loader(V_X, V_Y)
        # extracted_VX = get_extracted(feature_extractor, VX_loader, device)

        MMD_values = get_MMD_values_uneven(D_Xs, None, V_X, None)
        values_over_trials.append(MMD_values)

        MMD_values_hat = get_MMD_values_uneven(D_Xs, None, torch.cat(D_Xs), None)
        values_hat_over_trials.append(MMD_values_hat)

    results = {'values_over_trials': values_over_trials, 'values_hat_over_trials': values_hat_over_trials, 'N':N, 'size':size, 'n_trials': n_trials,
    'd':d, 'isHuber':not not_huber, 'heterogeneity': heterogeneity}
    if not_huber:
        exp_name = oj('not_huber', f'{dataset}_vs_{heterogeneity}-N{N} m{size} n_trials{n_trials}')
    else:
        exp_name = f'{dataset}_vs_{Q_dataset}-N{N} m{size} n_trials{n_trials}'
    save_results(baseline=baseline, exp_name=exp_name, **results)