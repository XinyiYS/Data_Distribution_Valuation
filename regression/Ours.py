import sys
sys.path.append("/home/xinyi/maplecg_nfs/codebase/Data_Distribution_Valuation")

import torch
from torch import optim
import torch.nn as nn
from torch.autograd import Variable

from mmd import rbf_mmd2


class GeneratorNet(nn.Module):
    def __init__(self, generator_nfilters, output_dim=10):
        super(GeneratorNet, self).__init__()

        layers = [nn.Linear(generator_nfilters, output_dim),
                # nn.LeakyReLU(0.2, inplace=True),
                # nn.Linear(generator_nfilters, output_dim),
                nn.Sigmoid()
        ]
        self.main = nn.Sequential(*layers)

    def forward(self, input):
        return self.main(input)

class DiscriminatorNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DiscriminatorNet, self).__init__()

        layers = [nn.Linear(input_dim, input_dim),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(input_dim, output_dim),
                # nn.BatchNorm1d(output_dim, affine=False)
        ]
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        output = self.main(x)
        return output.view(output.size(0), -1)

def get_reference_(dataloader, size=10000, dataset='CaliH', epochs=5, netD_output_dim=16, use_GMM=False, device=torch.device('cuda')):
    
    if use_GMM:
        print("Using GMM the parametric family.")
        from _Ours_utils import _get_mixture_dictionary
        mixtures =  _get_mixture_dictionary(dataset)

        data = torch.cat([x for x, y in dataloader])

        curr_MMD2, curr_generated = float('inf'), None
        for mixture in tqdm(mixtures, desc=f"Finding the best mixture out of {len(mixtures)} mixtures."):
            generated = torch.from_numpy(mixture.sample(len(data))[0]).float() # the mixture.sample returns [generated, component_labels]
            mmd2 = rbf_mmd2(data, generated)
            if mmd2 < curr_MMD2:
                curr_MMD2 = mmd2
                curr_generated = generated
        
        return curr_generated.to(device), None, None

    else:
        print("Using KDE method for the parametric family.")
        # KDE method
        data = torch.cat([x for x, y in dataloader]).detach().cpu().numpy()

        from sklearn.neighbors import KernelDensity as gaussian_kde
        kernel = gaussian_kde().fit(data)
        generated = kernel.sample(max(len(data), size), random_state=1234)
        
        return torch.from_numpy(generated), None, None

def get_reference(dataloader, size=10000, epochs=5, netD_output_dim=16):
    
    real, _ = next(iter(dataloader))

    netG = GeneratorNet(generator_nfilters=options.n_filters, output_dim=real.shape[1])
    netD = DiscriminatorNet(input_dim=real.shape[1], output_dim=netD_output_dim)
    noise = torch.FloatTensor(options.batch_size, options.n_filters)

    if options.cuda:
        netD.cuda()
        netG.cuda()
        noise = noise.cuda()

    # setup optimizers
    optimizerD = optim.Adam(netD.parameters(), lr=1e-4, betas=(0.5, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=1e-4, betas=(0.5, 0.999))

    Diters = 10
    Giters = 1

    sigma_list = [1, 2, 5, 10]
    for epoch in tqdm(range(epochs), desc="Training GAN."):
        for i, (real_cpu, _) in enumerate(dataloader):
            
            netD.zero_grad()
            netG.zero_grad()
            # get real batch
            if options.cuda:
                real_cpu = real_cpu.cuda()
            real = Variable(real_cpu)

            for Diter in range(Diters):
                # generate fake
                noise.normal_(0, 1)
                with torch.no_grad():
                    fake = netG(Variable(noise))[:real.size(0)]

                # joint to have common batch-norm transform
                outputs = netD(torch.cat([real, fake], dim=0))
                output_real = outputs[:real.size(0)]
                output_fake = outputs[real.size(0):]
                # MMD2 = mix_rbf_mmd2(output_real, output_fake, sigma_list)

                MMD2 = rbf_mmd2(output_real, output_fake, sigma_list) # allow unequal size
                (- MMD2).backward() # directly minimizing MMD2 instead of MMD for better stability
                # MMD = torch.sqrt(MMD2)                
                # neg_MMD = - MMD
                # neg_MMD.backward() 

                optimizerD.step()
            
            for Giter in range(Giters):
                # generate fake
                noise.normal_(0, 1)
                fake = netG(Variable(noise))[:real.size(0)]

                # joint to have common batch-norm transform
                outputs = netD(torch.cat([real, fake], dim=0))
                output_real = outputs[:real.size(0)]
                output_fake = outputs[real.size(0):]                
                # MMD2 = mix_rbf_mmd2(output_real, output_fake, sigma_list)
                MMD2 = rbf_mmd2(output_real, output_fake, sigma_list) # allow unequal size

                # MMD = torch.sqrt(MMD2)
                MMD2.backward() # directly minimizing MMD2 instead of MMD for better stability
                
                optimizerG.step()
        print(f"epoch {epoch}: MMD2 {MMD2}")

    netG.eval()
    generated_reference = []
    with torch.no_grad():
        for _ in range(size // options.batch_size + 1):
            noise.normal_(0, 1)
            generated_reference.append(netG(Variable(noise)))
    generated_reference = torch.cat(generated_reference)
    print(generated_reference.shape)

    return generated_reference[:size], netD, netG

from mmd import get_MMD_values_uneven

def get_MMD_values(D_Xs, D_Ys, V_X, V_Y, netD, sigma_list=[1, 2, 5, 10], device=torch.device('cuda')):
    results = []
    
    if netD:
        netD = netD.to(device)
        netD.eval()
    
    if isinstance(V_X, np.ndarray):
        V_X = torch.from_numpy(V_X)

    V_X = V_X.to(device)
    with torch.no_grad():
        for D_X in D_Xs:
            D_X = D_X.to(device)
            '''
            if len(D_X) >= len(V_X):
                min_len = min(len(D_X), len(V_X))
                MMD2 = mix_rbf_mmd2(D_X[:min_len].cuda(), V_X[:min_len].cuda(), sigma_list)
            else:
                D_X_rep = torch.cat([deepcopy(D_X) for _ in range(len(V_X)//len(D_X // 4))])
                min_len = min(len(D_X_rep), len(V_X))
                MMD2 = mix_rbf_mmd2(D_X_rep[:min_len].cuda(), V_X[:min_len].cuda(), sigma_list)
                print('---- Using multiple D_Xs ---- ')
            '''
            # min_len = min(len(D_X), len(V_X))
            # outputs = netD(torch.cat([V_X[:min_len].cuda(), D_X[:min_len].cuda()], dim = 0))
            # output_real = outputs[:min_len]
            # output_fake = outputs[min_len:]
            # MMD2 = mix_rbf_mmd2(output_real, output_fake, sigma_list)
            if netD:
                outputs = netD(torch.cat([V_X, D_X], dim = 0))
            else:
                outputs = torch.cat([V_X, D_X], dim=0)

            output_real = outputs[:len(V_X)]
            output_fake = outputs[len(V_X):]

            MMD2 = rbf_mmd2(output_real, output_fake, sigma_list) # allow unequal size
            results.append(-torch.sqrt(max(1e-6, MMD2)).item())

    return results


from copy import deepcopy
from utils import cwd, set_deterministic, save_results

from data_utils import _get_loader
import numpy as np
import torch

from reg_data_utils import assign_data
from os.path import join as oj

from tqdm import tqdm
import argparse

from scipy.stats import pearsonr, spearmanr

baseline = 'Ours'

class options:
    cuda = True
    batch_size = 256
    n_filters = 16
    epochs = 50

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

    values_over_trials, values_hat_over_trials = [], []
    values_hat_no_gen_over_trials = []
    values_mmd2_over_trials, values_hat_mmd2_over_trials = [], []
    for _ in tqdm(range(n_trials), desc =f'A total of {n_trials} trials.'):
        # raw data
        D_Xs, D_Ys, V_X, V_Y = assign_data(N, size, dataset, Q_dataset)

        loader = _get_loader(torch.cat(D_Xs), torch.cat(D_Ys), batch_size=options.batch_size)
        reference, netD, _ = get_reference_(loader, dataset=dataset, epochs=options.epochs, netD_output_dim=V_X.shape[1], device=device, use_GMM=use_GMM)

        # MMD_values = get_MMD_values(D_Xs, D_Ys, V_X, V_Y, netD, device=device)
        MMD_values = get_MMD_values_uneven(D_Xs, None, V_X, None, netD, device=device, batch_size=options.batch_size)
        print("MMD values:", MMD_values)
        values_over_trials.append(MMD_values)

        MMD2_values = get_MMD_values_uneven(D_Xs, None, V_X, None, netD, device=device, squared=True, batch_size=options.batch_size)
        values_mmd2_over_trials.append(MMD2_values)

        combined_reference = torch.cat([reference, torch.cat(D_Xs).to(device)]).float()
        # print(f" ---- combined reference shape: {combined_reference.shape} ---- ")
        MMD_values_hat = get_MMD_values_uneven(D_Xs, None, combined_reference, None, netD, device=device, batch_size=options.batch_size)
        values_hat_over_trials.append(MMD_values_hat)

        MMD_values_hat_no_gen = get_MMD_values_uneven(D_Xs, None, torch.cat(D_Xs).to(device), None, None, device=device, batch_size=options.batch_size)
        values_hat_no_gen_over_trials.append(MMD_values_hat_no_gen)

        MMD2_values_hat = get_MMD_values_uneven(D_Xs, None, combined_reference, None, netD, device=device, squared=True, batch_size=options.batch_size)
        values_hat_mmd2_over_trials.append(MMD2_values_hat)

    # Ours no gen
    results = {'values_over_trials': values_over_trials, 'values_hat_over_trials': values_hat_no_gen_over_trials, 
               'N':N, 'size':size, 'n_trials': n_trials, 'use_GMM': use_GMM}
    save_results(baseline=baseline, exp_name=oj('regression', f'{dataset}_vs_{Q_dataset}-N{N} m{size} n_trials{n_trials}'), **results)

    # Ours with gen only
    results = {'values_over_trials': values_hat_over_trials, 'values_hat_over_trials': values_hat_over_trials, 
               'N':N, 'size':size, 'n_trials': n_trials, 'use_GMM': use_GMM}
    baseline = baseline + "_GMM" if use_GMM else baseline +  '_KDE'
    save_results(baseline=baseline, exp_name=oj('regression', f'{dataset}_vs_{Q_dataset}-N{N} m{size} n_trials{n_trials}'), **results)


    # For MMD squared w.r.t. half mix reference
    results = {'values_over_trials': values_mmd2_over_trials, 'values_hat_over_trials': values_hat_mmd2_over_trials,
            'N':N, 'size':size, 'n_trials': n_trials, 'use_GMM': use_GMM}
    # save_results(baseline='MMD_sq_half_mix', exp_name=exp_name, **results)
    save_results(baseline='MMD_sq_half_mix', exp_name=oj('regression', f'{dataset}_vs_{Q_dataset}-N{N} m{size} n_trials{n_trials}'), **results)
