import os
from os.path import join as oj

from contextlib import contextmanager
@contextmanager
def cwd(path):
    oldpwd=os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(oldpwd)


def set_up_plotting(use_latex=True):

    from distutils.spawn import find_executable

    import seaborn as sns; sns.set_theme()
    import matplotlib.pyplot as plt

    LABEL_FONTSIZE = 24
    MARKER_SIZE = 10
    AXIS_FONTSIZE = 26
    TITLE_FONTSIZE= 26
    LINEWIDTH = 6

    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]

    # plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('figure', titlesize=TITLE_FONTSIZE)     # fontsize of the axes title
    plt.rc('axes', titlesize=TITLE_FONTSIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=AXIS_FONTSIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=LABEL_FONTSIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=LABEL_FONTSIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=LABEL_FONTSIZE)    # legend fontsize
    plt.rc('lines', markersize=MARKER_SIZE)  # fontsize of the figure title
    plt.rc('lines', linewidth=LINEWIDTH)  # fontsize of the figure title
    plt.rc('font', weight='bold') # set bold fonts


    if use_latex and find_executable('latex'): 
        # print("latex installed, using latex for matplotlib")
        plt.rcParams['text.usetex'] = True
    else:
        plt.rcParams['text.usetex'] = False

    return plt



import pandas as pd
def get_mean_se_df(correlation_df, name):

    mean_se_results = {}
    for col in correlation_df:
        avg, se = correlation_df[col].mean(), correlation_df[col].sem()
        mean_se_results[col] = f"{avg:.2} ({se:.2})"
    mean_se_df = pd.DataFrame(mean_se_results, index=[name])
    return mean_se_df

# import scipy.io
import numpy as np
import pandas as pd
# import mat73

# def mat_to_np(filename):
    
#     try:
#         mat = scipy.io.loadmat(filename)
#     except:  
#         mat = mat73.loadmat(filename)


#     X, y = mat['X'], mat['y']
#     dataset_name = filename.replace('.mat', '')
#     np.savez(f'{dataset_name}_count-{len(X)}_dim-{len(X[0])}.npz', X=X, y=y)
    
#     return

# with cwd('datasets'):
    # for file in os.listdir():
        # if file.endswith(".mat"):
            # mat_to_np(file)


def load_data(filename):
    
    data_dict = np.load(filename)
    X, y = data_dict['X'], data_dict['y']

    return X, y

# with cwd('datasets'):
    # X, y = load_data('annthyroid_count-7200_dim-6.npz')


# Compute MMD (maximum mean discrepancy) using numpy and scikit-learn.
import numpy as np
from sklearn import metrics

def mmd_linear(X, Y):
    """MMD using linear kernel (i.e., k(x,y) = <x,y>)
    Note that this is not the original linear MMD, only the reformulated and faster version.
    The original version is:
        def mmd_linear(X, Y):
            XX = np.dot(X, X.T)
            YY = np.dot(Y, Y.T)
            XY = np.dot(X, Y.T)
            return XX.mean() + YY.mean() - 2 * XY.mean()
    Arguments:
        X {[n_sample1, dim]} -- [X matrix]
        Y {[n_sample2, dim]} -- [Y matrix]
    Returns:
        [scalar] -- [MMD value]
    """
    delta = X.mean(0) - Y.mean(0)
    return delta.dot(delta.T)


def mmd_rbf(X, Y, gamma=1.0):
    """MMD using rbf (gaussian) kernel (i.e., k(x,y) = exp(-gamma * ||x-y||^2 / 2))
    Arguments:
        X {[n_sample1, dim]} -- [X matrix]
        Y {[n_sample2, dim]} -- [Y matrix]
    Keyword Arguments:
        gamma {float} -- [kernel parameter] (default: {1.0})
    Returns:
        [scalar] -- [MMD value]
    """
    XX = metrics.pairwise.rbf_kernel(X, X, gamma)
    YY = metrics.pairwise.rbf_kernel(Y, Y, gamma)
    XY = metrics.pairwise.rbf_kernel(X, Y, gamma)
    return XX.mean() + YY.mean() - 2 * XY.mean()


def mmd_poly(X, Y, degree=2, gamma=1, coef0=0):
    """MMD using polynomial kernel (i.e., k(x,y) = (gamma <X, Y> + coef0)^degree)
    Arguments:
        X {[n_sample1, dim]} -- [X matrix]
        Y {[n_sample2, dim]} -- [Y matrix]
    Keyword Arguments:
        degree {int} -- [degree] (default: {2})
        gamma {int} -- [gamma] (default: {1})
        coef0 {int} -- [constant item] (default: {0})
    Returns:
        [scalar] -- [MMD value]
    """
    XX = metrics.pairwise.polynomial_kernel(X, X, degree, gamma, coef0)
    YY = metrics.pairwise.polynomial_kernel(Y, Y, degree, gamma, coef0)
    XY = metrics.pairwise.polynomial_kernel(X, Y, degree, gamma, coef0)
    return XX.mean() + YY.mean() - 2 * XY.mean()

# a = np.arange(1, 10).reshape(3, 3)
# b = [[7, 6, 5], [4, 3, 2], [1, 1, 8], [0, 2, 5]]
# b = np.array(b)
# print(a)
# print(b)
# print(mmd_linear(a, b))  # 6.0
# print(mmd_rbf(a, b))  # 0.5822
# print(mmd_poly(a, b))  # 2436.5


import numpy as np
import random
import torch

def set_deterministic(seed=1234):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    return

from tqdm import tqdm

def get_trained_feature_extractor(model, train_loader, test_loader=None, device=torch.device('cuda'), epochs=50, load_path=''):

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, betas=(0.5, 0.999))
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    loss_fn = torch.nn.CrossEntropyLoss()

    model.train()
    model = model.to(device)
    for epoch in tqdm(range(int(epochs)), desc=f"Training feature extractor for {epochs} epochs."):
        for i, (batch_data, batch_target) in enumerate(train_loader):

            batch_data, batch_target = batch_data.to(device), batch_target.to(device)
            optimizer.zero_grad()
            ce_loss = loss_fn(model(batch_data), batch_target)
            ce_loss.backward()
            optimizer.step()

    if test_loader:
        model.eval()
        correct = 0
        total = 0
        loss = 0
        with torch.no_grad():
            for i, (batch_data, batch_target) in enumerate(test_loader):
                batch_data, batch_target = batch_data.to(device), batch_target.to(device)
                outputs = model(batch_data)

                loss += loss_fn(outputs, batch_target)
                correct += (torch.max(outputs, 1)[1].view(batch_target.size()).data == batch_target.data).sum()
                total += len(batch_target)

        accuracy = correct.float() / total
        print(f"Performance of trained feature extractor: test loss: {loss}, correct count: {correct}, total count: {total}, accuracy: {accuracy}.")
    return model

def get_accuracy(model, test_loader, device=torch.device('cuda')):

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for i, (batch_data, batch_target) in enumerate(test_loader):
            batch_data, batch_target = batch_data.to(device), batch_target.to(device)
            outputs = model(batch_data)

            correct += (torch.max(outputs, 1)[1].view(batch_target.size()).data == batch_target.data).sum()
            total += len(batch_target)
    
    if total != 0:
        accuracy =  correct * 1.0 / total
    else:
        accuracy = torch.tensor(0)
    # print(correct, total, accuracy)
    return accuracy


def save_results(baseline, exp_name='', **kwargs):
    results_dir = oj('results', exp_name, baseline)
    os.makedirs(results_dir, exist_ok=True)
    with cwd(results_dir):
        np.savez('values.npz', **kwargs)
    return


import torch
import torch.nn as nn
import torch.nn.functional as F
    

# for MNIST 32*32
class CNN_Net_32(nn.Module):

    def __init__(self, device=None):
        super(CNN_Net_32, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, 3, 1)
        self.conv2 = nn.Conv2d(64, 16, 7, 1)
        self.fc1 = nn.Linear(4 * 4 * 16, 200)
        self.fc2 = nn.Linear(200, 10)

    def forward(self, x):        
        x = x.view(-1, 1, 32, 32)
        x = torch.tanh(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 16)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
        # return F.log_softmax(x, dim=1) # use log_softmax with NLLLoss, if not use crossentropy loss
    

import torch

class LogisticRegression(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegression, self).__init__()
        self.first_layer = torch.nn.Linear(input_dim, 16)
        self.fc = torch.nn.Linear(16, output_dim)

    def forward(self, x):
        return self.fc(torch.nn.functional.leaky_relu(self.first_layer(x)))
        # outputs = torch.sigmoid(self.linear(x))
        # return outputs


class LinearRegression(torch.nn.Module):
    def __init__(self, input_dim, output_dim=1):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)  

    def forward(self, x):
        out = self.linear(x)
        return out

class MLPRegression(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim=16):
        super(MLPRegression, self).__init__()
        self.first_layer = torch.nn.Linear(input_dim, hidden_dim)
        self.fc = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x):
        return self.fc(torch.nn.functional.leaky_relu(self.first_layer(x)))

        
 