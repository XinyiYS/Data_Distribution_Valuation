import random
import numpy as np

#importing required libraries..
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler  #for validation test
import torch
from torch.utils.data import DataLoader, TensorDataset

TRANSFORMS = {
    'MNIST': transforms.Compose([
                transforms.Pad(2),
                transforms.ToTensor(),
                transforms.Normalize((0.5,),(0.5,),),
            ]),
    'EMNIST': transforms.Compose([
                transforms.Pad(2),
                transforms.ToTensor(),
                transforms.Normalize((0.5,),(0.5,),),
            ]),
    'FaMNIST': transforms.Compose([
                transforms.Pad(2),
                transforms.ToTensor(),
                transforms.Normalize((0.5,),(0.5,),),
            ]),
    'CIFAR10': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]),
    'CIFAR100': transforms.Compose([
                #transforms.ToPILImage(),
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ToTensor(),
                transforms.Normalize((0.5070751592371323, 0.48654887331495095, 0.4409178433670343), (0.2673342858792401, 0.2564384629170883, 0.27615047132568404))
            ]),

}

def huber(Pstar_X, Pstar_Y, Q, epsilon, size=2000, labels=[]):

    '''
    A larger epsilon means a higher chance to get a data point from the outlier distribution Q.
    '''
    data = []
    for _ in range(size):
        if random.random() < epsilon:
            Q_index = np.random.choice(len(Q))
            Q_X = Q[Q_index]

            random_label_index = random.randint(0, len(labels)-1)
            random_label = labels[random_label_index]

            data.append((Q_X, random_label))
        else:
            P_index = np.random.choice(len(Pstar_X))
            data.append((Pstar_X[P_index], Pstar_Y[P_index]))
    return data


def non_huber(X_train, y_train, degree, size=2000, heterogeneity='normal', vendor_index=None, N=None, interpolate=True):

    if heterogeneity == 'normal':
        indices = torch.from_numpy(np.random.choice(len(X_train), size=size))
        X_train = X_train[indices]
        y_train = y_train[indices]

        X_train += torch.randn_like(X_train) * 1 * degree

    elif heterogeneity == 'classimbalance' or heterogeneity == 'classimbalance_inter': 
        # full P indices
        # indices = torch.from_numpy(np.random.choice(len(X_train), size=int(size * (1-degree))))
        # X_train_ = X_train[indices]
        # y_train_ = y_train[indices]
        n_classes = 10
        classes = list(range(np.linspace(2, n_classes, N, dtype='int')[vendor_index]))

        if heterogeneity == 'classimbalance':
            class_size = size // len(classes)
            class_indices = []
            for class_id in classes:
                this_class_indices = torch.nonzero(y_train == class_id).view(-1).tolist()
                random.shuffle(this_class_indices)
                class_indices.append(torch.tensor(this_class_indices)[:class_size])
            class_indices = torch.cat(class_indices)

        else: # interpolating among different classes with an additional half of the alloted size
            half_size = size // 2

            # 1st half
            class_size = half_size // len(classes)
            class_indices = []
            for class_id in classes:
                this_class_indices = torch.nonzero(y_train == class_id).view(-1).tolist()
                random.shuffle(this_class_indices)
                class_indices.append(torch.tensor(this_class_indices)[:class_size])
            class_indices = torch.cat(class_indices)

            # 2nd half
            full_class_subset_indices = torch.from_numpy(np.random.choice(len(X_train), size=half_size))
            class_indices = torch.cat([class_indices, full_class_subset_indices])

        X_train = X_train[class_indices]
        y_train = y_train[class_indices]
    
    elif heterogeneity == 'classpartition':
        raise NotImplementedError(f"Heterogeneity {heterogeneity} not implemented.")
    else:
        raise NotImplementedError(f"Heterogeneity {heterogeneity} not implemented.")
    
    return [(X, y) for X, y in zip(X_train, y_train)]



def assign_data(N, size, dataset='MNIST', Q_dataset='EMNIST', not_huber=False, heterogeneity='normal'):

    if dataset == 'MNIST':
        X_train, y_train, X_test, y_test = _get_MNIST32()

    elif dataset == 'CIFAR10':
        X_train, y_train, X_test, y_test = _get_CIFAR10()

    elif dataset == 'CreditCard':
        X_train, y_train, X_test, y_test = _get_credit_card()

    elif dataset == 'TON':
        X_train, y_train, X_test, y_test = _get_TON()

    else:
        raise NotImplementedError(f'{dataset} is not implemented.')

    labels = sorted(list(set((y_train.tolist()))))
    print(f"For P={dataset}, X_train shape is {X_train.shape}, y_train shape is {y_train.shape}, and labels are {labels}.")
    print(f"Data types: X_train {X_train.dtype} y_train {y_train.dtype} X_test {X_test.dtype} y_test {y_test.dtype}.")

    if not_huber:
        # There is no need to explicitly set up Q for not Huber
        pass
    else:
        # set up Q
        if dataset == 'MNIST':
            if Q_dataset == 'EMNIST':
                EMNIST_train = datasets.EMNIST('./datasets/emnist', split='letters', download=True, train=True, transform=TRANSFORMS['EMNIST'])
                train_loader = DataLoader(EMNIST_train, batch_size=len(EMNIST_train), shuffle=True)
                Q, _ = next(iter(train_loader))

            elif Q_dataset == 'FaMNIST':

                FaMNIST_train = datasets.FashionMNIST('./datasets/F_MNIST_data', download=True, train=True, transform=TRANSFORMS['FaMNIST'])
                train_loader = DataLoader(FaMNIST_train, batch_size=len(FaMNIST_train), shuffle=True)
                Q, _ = next(iter(train_loader))

            elif Q_dataset == 'normal':
                Q = torch.randn(size=(size*2*N, 1, 32, 32))

            else:
                raise NotImplementedError(f'Q = {Q_dataset} is not implemented yet.')

        elif dataset == 'CIFAR10':
            if Q_dataset == 'CIFAR100':        
                cifar100_training = datasets.CIFAR100(root='./datasets/cifar100', train=True, download=True, transform=TRANSFORMS['CIFAR100'])
                cifar100_training_loader = DataLoader(cifar100_training, shuffle=True, batch_size=50000)
                Q, _ = next(iter(cifar100_training_loader))

            else:
                raise NotImplementedError(f'Q = {Q_dataset} is not implemented yet.')            

        elif dataset == 'CreditCard':
            if Q_dataset == 'CreditCard':
                X = pd.read_csv('datasets/credit1-PCA_features.csv')
                Q = torch.from_numpy(X.values).float()

        elif dataset == 'TON':
            if Q_dataset == 'UGR16':
                X = pd.read_csv('datasets/ugr16/ugr16-PCA_features.csv')
                Q = torch.from_numpy(X.values).float()
        
        else:
            raise NotImplementedError(f'P = {dataset} is not implemented yet.')

    D_Xs, D_Ys = [], []
    for i in range(N):
        if not_huber:
            data = non_huber(X_train, y_train, degree = (i*1.0) / N, size=size, heterogeneity=heterogeneity, vendor_index=i, N=N)
        else:
            # As the index i increases, the epsilon_i increases, namely the outlier distribution Q has a higher impact.
            if dataset == 'CreditCard' or dataset == 'TON':
                # more outlier for these two datasets
                data = huber(X_train, y_train, Q, epsilon = (i*4.0) / N, size=size, labels=labels)
            else:
                # for MNIST and CIFAR10, this epsilon is sufficient
                data = huber(X_train, y_train, Q, epsilon = (i*1.0) / N, size=size, labels=labels)

        D_X = torch.stack([X for X, y in data])
        D_Y = torch.Tensor([y for X, y in data])

        D_Xs.append(D_X)
        D_Ys.append(D_Y)

    return D_Xs, D_Ys, X_test, y_test, labels


def _get_loader(X, y, batch_size=64, shuffle=True, drop_last=True, mode='cls'):
    tensor_x = torch.from_numpy(X) if isinstance(X, np.ndarray) else X
    tensor_y = torch.from_numpy(y) if isinstance(y, np.ndarray) else y

    if mode == 'cls':
        dataset = TensorDataset(tensor_x.float(), tensor_y.long()) # create your datset
    elif mode == 'reg':
        dataset = TensorDataset(tensor_x.float(), tensor_y.float()) # create your datset
    else:
        raise NotImplementedError(f"Learning mode for {mode} is not implemented. Only 'cls' or 'reg'.")
    # the drop_last=True is specifically for models with BatchNormLayer and when there is only a single data point in the loader, it creates an error
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last) # create your dataloader


def _get_MNIST32():
    MNIST_train = datasets.MNIST('./datasets/mnist', download=True, train=True, transform=TRANSFORMS['MNIST'])
    MNIST_test = datasets.MNIST('./datasets/mnist', download=True, train=False, transform=TRANSFORMS['MNIST'])

    train_loader = DataLoader(MNIST_train, batch_size=len(MNIST_train), shuffle=True)
    X_train, y_train = next(iter(train_loader))

    # train_images, train_labels = train_images.numpy(), train_labels.numpy()
    test_loader = DataLoader(MNIST_test, batch_size=len(MNIST_test), shuffle=True)
    X_test, y_test = next(iter(test_loader))

    return X_train.float(), y_train.long(), X_test.float(), y_test.long()

def _get_CIFAR10():
    CIFAR10_train = datasets.CIFAR10(root='./datasets/cifar10', train=True, download=True, transform=TRANSFORMS['CIFAR10'])
    CIFAR10_test = datasets.CIFAR10(root='./datasets/cifar10', train=False, download=True, transform=TRANSFORMS['CIFAR10'])

    train_loader = DataLoader(CIFAR10_train, batch_size=len(CIFAR10_train), shuffle=True)
    X_train, y_train = next(iter(train_loader))

    # train_images, train_labels = train_images.numpy(), train_labels.numpy()
    test_loader = DataLoader(CIFAR10_test, batch_size=len(CIFAR10_test), shuffle=True)
    X_test, y_test = next(iter(test_loader))
    # test_images, test_labels = test_images.numpy(), test_labels.numpy()
    return X_train.float(), y_train.long(), X_test.float(), y_test.long()


from sklearn.model_selection import train_test_split
from utils import cwd
import pandas as pd 
from sklearn.utils import shuffle

def _get_TON():
    X = pd.read_csv('datasets/ton/ton-features.csv')
    y = pd.read_csv('datasets/ton/ton-labels.csv')
    
    X = torch.from_numpy(X.values)
    y = torch.from_numpy(y.values).squeeze()
    X, y = shuffle(X, y, random_state=0)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1234)
    print("Train test shapes:", X_train.shape, y_train.shape, X_test.shape, y_test.shape)
    return X_train.float(), y_train.long(), X_test.float(), y_test.long()


def _get_credit_card():
    X = pd.read_csv('datasets/credit2-features.csv')
    y = pd.read_csv('datasets/credit2-labels.csv')

    # subsampling the nonpositive cases
    X_0 = X[y.values == 0]
    y_0 = y[y.values == 0]

    X_1 = X[y.values == 1.0]
    y_1 = y[y.values == 1.0]

    X_0, y_0 = shuffle(X_0, y_0, random_state=0)
    X_sub = np.concatenate([X_1, X_0[:len(X_1)]])
    y_sub = np.concatenate([y_1, y_0[:len(y_1)]])

    X = torch.from_numpy(X_sub)
    y = torch.from_numpy(y_sub).squeeze()
    X, y = shuffle(X, y, random_state=0)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1234)
    print("Train test shapes:", X_train.shape, y_train.shape, X_test.shape, y_test.shape)
    return X_train.float(), y_train.long(), X_test.float(), y_test.long()

'''
def show_images(x, n_images=10):
    x_numpy = x.cpu().numpy() * 0.5 + 0.5
    plt.figure(figsize=[15, 5])
    for i in range(min(n_images, x_numpy.shape[0])):
        plt.subplot(1, n_images, i + 1)
        plt.axis('off')
        if x_numpy.shape[1] == 1:
            plt.imshow(x_numpy[i, 0], vmin=0, vmax=1, cmap='gray')
        else:
            plt.imshow(numpy.einsum('cij->ijc', x_numpy[i]))

    plt.show()
'''
