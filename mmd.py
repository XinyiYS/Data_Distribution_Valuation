#!/usr/bin/env python
# encoding: utf-8


import torch

min_var_est = 1e-8

import numpy as np

################################################################################
# From DavinZ to enable unequal sizes of X, Y
################################################################################

def rbf_mmd2(X, Y, sigma_list=[1, 2, 5, 10], biased=True, device=torch.device('cuda')):
    """
    Computes squared MMD using a RBF kernel.
    
    Args:
        X, Y (Tensor): datasets that MMD is computed on
        sigma (float): lengthscale of the RBF kernel
        biased (bool): whether to compute a biased mean
        
    Return:
        MMD squared
    """

    if len(X.shape) > 2:
        X = X.view(len(X), -1)

    if len(Y.shape) > 2:
        Y = Y.view(len(Y), -1)

    X = X.to(device)
    Y = Y.to(device)
    
    XX = torch.matmul(X, X.T)
    XY = torch.matmul(X, Y.T)
    YY = torch.matmul(Y, Y.T)
    
    X_sqnorms = torch.diagonal(XX)
    Y_sqnorms = torch.diagonal(YY)
    
    assert len(sigma_list) > 0

    K_XYs, K_XXs, K_YYs = [], [], []
    for sigma in sigma_list:
        gamma = 1 / (2 * sigma**2)
        
        K_XY = torch.exp(-gamma * (
                -2 * XY + X_sqnorms[:, np.newaxis] + Y_sqnorms[np.newaxis, :]))
        K_XX = torch.exp(-gamma * (
                -2 * XX + X_sqnorms[:, np.newaxis] + X_sqnorms[np.newaxis, :]))
        K_YY = torch.exp(-gamma * (
                -2 * YY + Y_sqnorms[:, np.newaxis] + Y_sqnorms[np.newaxis, :]))
        
        K_XXs.append(K_XX)
        K_XYs.append(K_XY)
        K_YYs.append(K_YY)

    K_XY = torch.stack(K_XYs).sum(dim=0)
    K_XX = torch.stack(K_XXs).sum(dim=0)
    K_YY = torch.stack(K_YYs).sum(dim=0)
    if biased:
        mmd2 = K_XX.mean() + K_YY.mean() - 2 * K_XY.mean()
    else:
        m = K_XX.shape[0]
        n = K_YY.shape[0]

        mmd2 = ((K_XX.sum() - m) / (m * (m - 1))
              + (K_YY.sum() - n) / (n * (n - 1))
              - 2 * K_XY.mean())
    return mmd2

from sklearn.utils import shuffle

def batched_rbf_mmd2(X, Y, sigma_list=[1, 2, 5, 10], biased=True, device=torch.device('cuda'), batch_size=1024):
    
    X, Y = shuffle(X, Y, random_state=1234)

    X_batches = torch.split(X, batch_size)
    Y_batches = torch.split(Y, batch_size)
    # print(f"Number of batches: {len(X_batches)}, {len(Y_batches)} at size {batch_size}")
    overall_mmd2 = 0
    for i, (x, y) in enumerate(zip(X_batches, Y_batches)):
        batch_mmd2 = rbf_mmd2(x, y, sigma_list=sigma_list, biased=True, device=device)
        overall_mmd2 += batch_mmd2 / batch_size

    return overall_mmd2


# Consider linear time MMD with a linear kernel:
# K(f(x), f(y)) = f(x)^Tf(y)
# h(z_i, z_j) = k(x_i, x_j) + k(y_i, y_j) - k(x_i, y_j) - k(x_j, y_i)
#             = [f(x_i) - f(y_i)]^T[f(x_j) - f(y_j)]
#
# f_of_X: batch_size * k
# f_of_Y: batch_size * k
def linear_mmd2(f_of_X, f_of_Y):
    loss = 0.0
    delta = f_of_X - f_of_Y
    loss = torch.mean((delta[:-1] * delta[1:]).sum(1))
    return loss


# Consider linear time MMD with a polynomial kernel:
# K(f(x), f(y)) = (alpha*f(x)^Tf(y) + c)^d
# f_of_X: batch_size * k
# f_of_Y: batch_size * k
def poly_mmd2(f_of_X, f_of_Y, d=2, alpha=1.0, c=2.0):
    K_XX = (alpha * (f_of_X[:-1] * f_of_X[1:]).sum(1) + c)
    K_XX_mean = torch.mean(K_XX.pow(d))

    K_YY = (alpha * (f_of_Y[:-1] * f_of_Y[1:]).sum(1) + c)
    K_YY_mean = torch.mean(K_YY.pow(d))

    K_XY = (alpha * (f_of_X[:-1] * f_of_Y[1:]).sum(1) + c)
    K_XY_mean = torch.mean(K_XY.pow(d))

    K_YX = (alpha * (f_of_Y[:-1] * f_of_X[1:]).sum(1) + c)
    K_YX_mean = torch.mean(K_YX.pow(d))

    return K_XX_mean + K_YY_mean - K_XY_mean - K_YX_mean


def _mix_rbf_kernel(X, Y, sigma_list=[1,2,5,10]):
    assert(X.size(0) == Y.size(0))
    m = X.size(0)

    Z = torch.cat((X, Y), 0)
    ZZT = torch.mm(Z, Z.t())
    diag_ZZT = torch.diag(ZZT).unsqueeze(1)
    Z_norm_sqr = diag_ZZT.expand_as(ZZT)
    exponent = Z_norm_sqr - 2 * ZZT + Z_norm_sqr.t()

    K = 0.0
    for sigma in sigma_list:
        gamma = 1.0 / (2 * sigma**2)
        K += torch.exp(-gamma * exponent)

    return K[:m, :m], K[:m, m:], K[m:, m:], len(sigma_list)


def mix_rbf_mmd2(X, Y, sigma_list=[1,2,5,10], biased=True):
    if len(X.shape) > 2:
        X = X.view(len(X), -1)

    if len(Y.shape) > 2:
        Y = Y.view(len(Y), -1)

    K_XX, K_XY, K_YY, d = _mix_rbf_kernel(X, Y, sigma_list)
    # return _mmd2(K_XX, K_XY, K_YY, const_diagonal=d, biased=biased)
    return _mmd2(K_XX, K_XY, K_YY, const_diagonal=False, biased=biased)


def mix_rbf_mmd2_and_ratio(X, Y, sigma_list, biased=True):
    K_XX, K_XY, K_YY, d = _mix_rbf_kernel(X, Y, sigma_list)
    # return _mmd2_and_ratio(K_XX, K_XY, K_YY, const_diagonal=d, biased=biased)
    return _mmd2_and_ratio(K_XX, K_XY, K_YY, const_diagonal=False, biased=biased)


################################################################################
# Helper functions to compute variances based on kernel matrices
################################################################################


def _mmd2(K_XX, K_XY, K_YY, const_diagonal=False, biased=False):
    m = K_XX.size(0)    # assume X, Y are same shape

    # Get the various sums of kernels that we'll use
    # Kts drop the diagonal, but we don't need to compute them explicitly
    if const_diagonal is not False:
        diag_X = diag_Y = const_diagonal
        sum_diag_X = sum_diag_Y = m * const_diagonal
    else:
        diag_X = torch.diag(K_XX)                       # (m,)
        diag_Y = torch.diag(K_YY)                       # (m,)
        sum_diag_X = torch.sum(diag_X)
        sum_diag_Y = torch.sum(diag_Y)

    Kt_XX_sums = K_XX.sum(dim=1) - diag_X             # \tilde{K}_XX * e = K_XX * e - diag_X
    Kt_YY_sums = K_YY.sum(dim=1) - diag_Y             # \tilde{K}_YY * e = K_YY * e - diag_Y
    K_XY_sums_0 = K_XY.sum(dim=0)                     # K_{XY}^T * e

    Kt_XX_sum = Kt_XX_sums.sum()                       # e^T * \tilde{K}_XX * e
    Kt_YY_sum = Kt_YY_sums.sum()                       # e^T * \tilde{K}_YY * e
    K_XY_sum = K_XY_sums_0.sum()                       # e^T * K_{XY} * e

    if biased:
        mmd2 = ((Kt_XX_sum + sum_diag_X) / (m * m)
            + (Kt_YY_sum + sum_diag_Y) / (m * m)
            - 2.0 * K_XY_sum / (m * m))
    else:
        mmd2 = (Kt_XX_sum / (m * (m - 1))
            + Kt_YY_sum / (m * (m - 1))
            - 2.0 * K_XY_sum / (m * m))

    return mmd2


def _mmd2_and_ratio(K_XX, K_XY, K_YY, const_diagonal=False, biased=False):
    mmd2, var_est = _mmd2_and_variance(K_XX, K_XY, K_YY, const_diagonal=const_diagonal, biased=biased)
    loss = mmd2 / torch.sqrt(torch.clamp(var_est, min=min_var_est))
    return loss, mmd2, var_est


def _mmd2_and_variance(K_XX, K_XY, K_YY, const_diagonal=False, biased=False):
    m = K_XX.size(0)    # assume X, Y are same shape

    # Get the various sums of kernels that we'll use
    # Kts drop the diagonal, but we don't need to compute them explicitly
    if const_diagonal is not False:
        diag_X = diag_Y = const_diagonal
        sum_diag_X = sum_diag_Y = m * const_diagonal
        sum_diag2_X = sum_diag2_Y = m * const_diagonal**2
    else:
        diag_X = torch.diag(K_XX)                       # (m,)
        diag_Y = torch.diag(K_YY)                       # (m,)
        sum_diag_X = torch.sum(diag_X)
        sum_diag_Y = torch.sum(diag_Y)
        sum_diag2_X = diag_X.dot(diag_X)
        sum_diag2_Y = diag_Y.dot(diag_Y)

    Kt_XX_sums = K_XX.sum(dim=1) - diag_X             # \tilde{K}_XX * e = K_XX * e - diag_X
    Kt_YY_sums = K_YY.sum(dim=1) - diag_Y             # \tilde{K}_YY * e = K_YY * e - diag_Y
    K_XY_sums_0 = K_XY.sum(dim=0)                     # K_{XY}^T * e
    K_XY_sums_1 = K_XY.sum(dim=1)                     # K_{XY} * e

    Kt_XX_sum = Kt_XX_sums.sum()                       # e^T * \tilde{K}_XX * e
    Kt_YY_sum = Kt_YY_sums.sum()                       # e^T * \tilde{K}_YY * e
    K_XY_sum = K_XY_sums_0.sum()                       # e^T * K_{XY} * e

    Kt_XX_2_sum = (K_XX ** 2).sum() - sum_diag2_X      # \| \tilde{K}_XX \|_F^2
    Kt_YY_2_sum = (K_YY ** 2).sum() - sum_diag2_Y      # \| \tilde{K}_YY \|_F^2
    K_XY_2_sum  = (K_XY ** 2).sum()                    # \| K_{XY} \|_F^2

    if biased:
        mmd2 = ((Kt_XX_sum + sum_diag_X) / (m * m)
            + (Kt_YY_sum + sum_diag_Y) / (m * m)
            - 2.0 * K_XY_sum / (m * m))
    else:
        mmd2 = (Kt_XX_sum / (m * (m - 1))
            + Kt_YY_sum / (m * (m - 1))
            - 2.0 * K_XY_sum / (m * m))

    var_est = (
        2.0 / (m**2 * (m - 1.0)**2) * (2 * Kt_XX_sums.dot(Kt_XX_sums) - Kt_XX_2_sum + 2 * Kt_YY_sums.dot(Kt_YY_sums) - Kt_YY_2_sum)
        - (4.0*m - 6.0) / (m**3 * (m - 1.0)**3) * (Kt_XX_sum**2 + Kt_YY_sum**2)
        + 4.0*(m - 2.0) / (m**3 * (m - 1.0)**2) * (K_XY_sums_1.dot(K_XY_sums_1) + K_XY_sums_0.dot(K_XY_sums_0))
        - 4.0*(m - 3.0) / (m**3 * (m - 1.0)**2) * (K_XY_2_sum) - (8 * m - 12) / (m**5 * (m - 1)) * K_XY_sum**2
        + 8.0 / (m**3 * (m - 1.0)) * (
            1.0 / m * (Kt_XX_sum + Kt_YY_sum) * K_XY_sum
            - Kt_XX_sums.dot(K_XY_sums_1)
            - Kt_YY_sums.dot(K_XY_sums_0))
        )
    return mmd2, var_est

import torch
from mmd import batched_rbf_mmd2

def get_MMD_values_uneven(D_Xs, D_Ys, V_X, V_Y, netD=None, device=torch.device('cuda'), sample_size=None, squared=False, batch_size=1024, sigma_list = [1, 2, 5, 10],):
    """
    Based on an implementation of MMD2 that enables different-sized inputs. 
    Also leverages the sigma list.
    """
    results = []
    V_X = V_X.to(device)

    rand_indx = torch.randperm(len(V_X))
    permuted_V_X = V_X[rand_indx] # permutation is used because later batching is applied AND possibly only a subset of V_X is used 

    if netD is None:        
        for D_X in D_Xs:

            # permutation is used because later batching is applied AND possibly only a subset of V_X is used 
            D_X = D_X[torch.randperm(len(D_X))]
            if sample_size is not None:
                permuted_V_X = permuted_V_X[:sample_size]
                D_X = D_X[:sample_size] 
    
            min_len = min(len(permuted_V_X), len(D_X))

            MMD2 = batched_rbf_mmd2(D_X[:min_len], permuted_V_X[:min_len], sigma_list, device=device, batch_size=batch_size) # use a batched version of rbf_mmd2 to avoid OOM error
            if squared:
                results.append(-MMD2.item())
            else:
                # take the square root
                results.append(-torch.sqrt(max(1e-6, MMD2)).item())

        return results      
    
    else: # netD is given and not None
        netD = netD.to(device)

        results = []
        for D_X in D_Xs:
            D_X = D_X.to(device)
            outputs = netD(torch.cat([permuted_V_X, D_X], dim=0))

            transformed_ref = outputs[:V_X.size(0)]
            transformed_D_X = outputs[V_X.size(0):]
            transformed_D_X = transformed_D_X[torch.randperm(len(transformed_D_X))] # permute it
            if sample_size is not None:
                transformed_ref = transformed_ref[:sample_size]
                transformed_D_X = transformed_D_X[:sample_size]
            
            min_len = min(len(transformed_ref), len(transformed_D_X))

            MMD2 = batched_rbf_mmd2(transformed_D_X[:min_len], transformed_ref[:min_len], sigma_list, device=device, batch_size=batch_size) # use a batched version of rbf_mmd2 to avoid OOM error
            if squared:
                results.append(-MMD2.item())
            else:
                # take the square root
                results.append(-torch.sqrt(max(1e-6, MMD2)).item())
        return results