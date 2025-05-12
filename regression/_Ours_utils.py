import numpy as np
from sklearn.utils import shuffle, resample
from sklearn.mixture import GaussianMixture
from tqdm import tqdm
import torch

def get_trained_regressor(model, train_loader, test_loader=None, epochs=20, device=torch.device('cuda')):

    # optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, betas=(0.5, 0.999))
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
    loss_fn = torch.nn.MSELoss()

    model.train()
    model = model.to(device)
    for epoch in tqdm(range(int(epochs)), desc=f"Training regressor for {epochs} epochs."):
        for i, (batch_data, batch_target) in enumerate(train_loader):

            batch_data, batch_target = batch_data.to(device), batch_target.to(device)
            optimizer.zero_grad()
            mse_loss = loss_fn(model(batch_data), batch_target)
            mse_loss.backward()
            optimizer.step()

    if test_loader:
        model.eval()
        total = 0
        loss = 0
        with torch.no_grad():
            for i, (batch_data, batch_target) in enumerate(test_loader):
                batch_data, batch_target = batch_data.to(device), batch_target.to(device)
                outputs = model(batch_data)

                loss += loss_fn(outputs, batch_target)
                total += len(batch_target)

        print(f"Performance of trained regressor: test MSE loss: {loss}, total count: {total}.")
    return model

def _get_mixture_dictionary(dataset='CaliH'):

    if dataset =='CaliH':
        from reg_data_utils import _get_CaliH, _get_KingH
        P_data, _ = _get_CaliH()
        Q_data, _ = _get_KingH()


    elif dataset == 'Census15':
        from reg_data_utils import _get_census

        P_data, _ = _get_census(15)
        Q_data, _ = _get_census(17)

    else:
        raise NotImplementedError(f"P={dataset} is not implemented.")

    true_mixture = GaussianMixture(n_components=1).fit(P_data)

    mixtures = [true_mixture]

    for pct in tqdm(np.arange(0.1, 1, 0.1), desc=f"Fitting P-Q mixture from the range {np.arange(0.1, 1, 0.1)}"):
        m = min(len(P_data), len(Q_data))

        P_data_sub = resample(P_data, n_samples=int((1-pct)*m))
        Q_data_sub = resample(Q_data, n_samples=int(pct*m))
        data = np.concatenate((P_data_sub, Q_data_sub), axis=0)
        data = shuffle(data)

        mixture = GaussianMixture(n_components=2).fit(data)
        mixtures.append(mixture)

    return mixtures