from os.path import join as oj
import pandas as pd
import torch
import random
import numpy as np


from utils import cwd

def _get_CaliH():

    with cwd(oj("./regression")):
        features = pd.read_csv("CaliH-NN_features.csv")
        labels = pd.read_csv("CaliH-labels.csv")

    X_train = torch.from_numpy(features.values)
    y_train = torch.from_numpy(labels.values)
    print(f'California housing dataset: X_train shape {X_train.shape}')

    return X_train, y_train


def _get_KingH():
    with cwd(oj("./regression")):

        features = pd.read_csv("KingH-NN_features.csv")
        labels = pd.read_csv("KingH-labels.csv")

    X_train = torch.from_numpy(features.values)
    y_train = torch.from_numpy(labels.values)
    print(f'King County Housing dataset: X_train shape {X_train.shape}')
    return X_train, y_train


def _get_FaceA():

    with cwd(oj("./")):

        features = pd.read_csv("face_age-CNN_features.csv")
        labels = pd.read_csv("face_age-labels.csv")

    X_train = torch.from_numpy(features.values)
    y_train = torch.from_numpy(labels.values)
    print(f'Face Age dataset: X_train shape {X_train.shape}')

    return X_train, y_train


def _get_census(year=15):
    if year == 17:
        with cwd(oj("./regression")):

            features = pd.read_csv("USCensus-2017-NN_features.csv")
            labels = pd.read_csv("USCensus-2017-labels.csv")

        X_train = torch.from_numpy(features.values)
        y_train = torch.from_numpy(labels.values)
        print(f'US Census income 2017 dataset: X_train shape {X_train.shape}')
    elif year == 15:
        with cwd(oj("./regression")):
            features = pd.read_csv("USCensus-2015-NN_features.csv")
            labels = pd.read_csv("USCensus-2015-labels.csv")

        X_train = torch.from_numpy(features.values)
        y_train = torch.from_numpy(labels.values)
        print(f'US Census income 2015 dataset: X_train shape {X_train.shape}')

    return X_train, y_train



def huber_regression(Pstar_X, Pstar_Y, Q_X, Q_Y, epsilon, size=2000):
    D_X, D_Y = [], []
    for _ in range(size):
        if random.random() < epsilon:
            Q_index = np.random.choice(len(Q_X))
            
            D_X.append(Q_X[Q_index])
            D_Y.append(Q_Y[Q_index])
        else:
            P_index = np.random.choice(len(Pstar_X))

            D_X.append(Pstar_X[P_index])
            D_Y.append(Pstar_Y[P_index])

    return torch.stack(D_X), torch.stack(D_Y)



from sklearn.model_selection import train_test_split

def assign_data(N, size, dataset='CaliH', Q_dataset='KingH', factor=10):

    if dataset == 'CaliH':
        X, y = _get_CaliH()

    elif dataset == 'Census15':
        X, y = _get_census(15)
    else:
        raise NotImplementedError(f"P = {dataset} not implemented.")

    X = X.to(torch.float32)
    y = y.to(torch.float32)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1234)
    print("Train test shapes:", X_train.shape, y_train.shape, X_test.shape, y_test.shape)

    if Q_dataset == 'KingH':
        Q_X, Q_y = _get_KingH()
    elif Q_dataset == 'Census17':
        Q_X, Q_y = _get_census(17)
    else:
        raise NotImplementedError(f"Q = {Q_dataset} not implemented.")
    
    Q_X = Q_X.to(torch.float32)
    Q_y = Q_y.to(torch.float32)

    D_Xs, D_Ys = [], []
    for i in range(N):
        D_X, D_Y = huber_regression(X_train, y_train, Q_X, Q_y, epsilon = (i*1.0) / (factor*N), size=size)
        D_Xs.append(D_X)
        D_Ys.append(D_Y)

    return D_Xs, D_Ys, X_test, y_test





