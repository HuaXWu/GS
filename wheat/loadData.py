import torch
import numpy as np
from torch.utils import data
from sklearn.model_selection import train_test_split
import pandas as pd
from scipy.stats import stats


def load_data(itrait):
    # X = pd.read_csv("/root/autodl-tmp/data/wheat.X", header=None, sep='\s+', dtype=np.float32)
    # Y = pd.read_csv("/root/autodl-tmp/data/wheat.Y", header=None, sep='\s+', dtype=np.float32)
    X = pd.read_csv("/home/WuHX/Deep-Learning-for-GS/data/wheat/wheat.X", header=None, sep='\s+', dtype=np.float32)
    Y = pd.read_csv("/home/WuHX/Deep-Learning-for-GS/data/wheat/wheat.Y", header=None, sep='\s+', dtype=np.float32)
    X_train, X_test, y_train, y_test = train_test_split(X, Y[itrait], test_size=0.2, random_state=11)
    # X_train = X_train.iloc[:, 1:]
    # X_test = X_test.iloc[:, 1:]
    return snp_preselection(X_train, X_test, y_train, y_test)


def snp_preselection(X_train, X_test, y_train, y_test):
    pals = []
    for i in range(X_train.shape[1]):
        b, intercept, r_value, p_value, std_err = stats.linregress(X_train[i], y_train)
        pals.append(-np.log10(p_value))
    pals = np.array(pals)
    min_P_value = 1.111
    snp_list = np.nonzero(pals > min_P_value)
    X_train = X_train[X_train.columns[snp_list]]
    X_test = X_test[X_test.columns[snp_list]]
    return toTensorLoader(X_train, X_test, y_train, y_test)


def toTensorLoader(X_train, X_test, y_train, y_test):
    # step 1 to numpy
    X_train = torch.from_numpy(np.array(X_train))
    y_train = torch.from_numpy(np.array(y_train)).type(torch.FloatTensor)
    X_test = torch.from_numpy(np.array(X_test))
    y_test = torch.from_numpy(np.array(y_test)).type(torch.FloatTensor)

    # step 2 to reshape 2d
    X_train = X_train.reshape(479, 18, 18)
    y_train = y_train.reshape(479, -1)
    X_test = X_test.reshape(120, 18, 18)
    y_test = y_test.reshape(120, -1)

    # step 3 to 3d
    X_train = torch.unsqueeze(X_train, dim=1).type(torch.FloatTensor)
    X_test = torch.unsqueeze(X_test, dim=1).type(torch.FloatTensor)

    print("X_train shape is {}, X_test shape is {}"
          .format(X_train.shape, X_test.shape))

    # step to tensor
    train_tensor = data.TensorDataset(X_train, y_train)
    test_tensor = data.TensorDataset(X_test, y_test)

    # step 5 to dataloader
    train_loader_test = data.DataLoader(train_tensor, batch_size=120, shuffle=True)
    test_loader_test = data.DataLoader(test_tensor, batch_size=40, shuffle=False)

    return train_loader_test, test_loader_test


