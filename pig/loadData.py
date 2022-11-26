import json

import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.utils import data
from torchvision import transforms
import feather


def load_data(random_state, label_col_name):
    X = np.load("/root/autodl-tmp/data/z_data/bf_snp_01.npy")
    Y = feather.read_dataframe("/root/autodl-tmp/data/z_data/label.feather")

    # with open("class_indices.json", 'r+') as f:
    #     dict = json.load(f)
    # for i in range(8, 16):
    #     Y[label_col_name].replace(i, dict.get(str(i)), inplace=True)
    # label10 = Y[Y[label_col_name] == dict.get(str(10))]
    # label11 = Y[Y[label_col_name] == dict.get(str(11))]
    # label12 = Y[Y[label_col_name] == dict.get(str(12))]
    # x_10 = X[label10.index]
    # x_11 = X[label11.index]
    # x_12 = X[label12.index]
    # X = np.concatenate((x_10, x_11, x_12), axis=0)
    # Y = pd.concat([label10, label11, label12])
    x_train_test, x_valid, y_train_test, y_valid = train_test_split(X, Y, random_state=random_state, test_size=0.2)
    x_train, x_test, y_train, y_test = train_test_split(x_train_test, y_train_test, random_state=random_state,
                                                        test_size=0.2)
    print("X_train shape is {}, X_test shape is {}, x_valid shape is {}"
          .format(x_train.shape, x_test.shape, x_valid.shape))
    # y_train = np.array(y_train[label_col_name])
    # y_test = np.array(y_test[label_col_name])
    # y_valid = np.array(y_valid[label_col_name])
    return x_train, x_test, x_valid, y_train, y_test, y_valid


def to_tensor_loader(X_train, X_test, y_train, y_test, ):
    # step 1 to tensor
    # X_train = X_train.astype(np.int64)
    X_train = torch.from_numpy(np.array(X_train))
    y_train = torch.from_numpy(np.array(y_train))

    # X_test = X_test.astype(np.int64)
    X_test = torch.from_numpy(np.array(X_test))
    y_test = torch.from_numpy(np.array(y_test))

    print("X_train shape is {}, X_test shape is {}"
          .format(X_train.shape, X_test.shape))

    # step 4 to tensorDataset
    X_train = X_train.type(torch.float32)
    y_train = y_train.type(torch.float32)
    # X_train = transforms.ToPILImage()(X_train)
    # X_train = transforms.Resize(224)(X_train)
    # X_train = transforms.ToTensor()(X_train)
    X_train = transforms.Normalize([0.4337, 0.2384, 0.3279], [0.4956, 0.4261, 0.4695])(X_train)
    # data_transform = {
    #     "train": transforms.Compose([transforms.ToPILImage(),
    #                                  transforms.Resize(224),
    #                                  transforms.PILToTensor(),
    #                                  transforms.Normalize([0.4337, 0.2384, 0.3279], [0.4956, 0.4261, 0.4695])]),
    #     "val": transforms.Compose([transforms.ToPILImage(),
    #                                transforms.Resize(224),
    #                                transforms.PILToTensor(),
    #                                transforms.Normalize([0.4337, 0.2384, 0.3279], [0.4956, 0.4261, 0.4695])])
    #
    # }
    train_tensor = data.TensorDataset(X_train, y_train)
    X_test = X_test.type(torch.float32)
    y_test = y_test.type(torch.float32)
    # X_test = transforms.ToPILImage(X_test)
    # X_test = transforms.Resize(224)(X_test)
    # X_test = transforms.ToTensor()(X_test)
    X_test = transforms.Normalize([0.4337, 0.2384, 0.3279], [0.4956, 0.4261, 0.4695])(X_test)
    test_tensor = data.TensorDataset(X_test, y_test)

    # step 5 to dataloader
    train_loader = data.DataLoader(train_tensor, batch_size=8, shuffle=True, num_workers=8)
    test_loader = data.DataLoader(test_tensor, batch_size=6, shuffle=False, num_workers=8)
    return train_loader, test_loader


def load_valid_data(X_valid, y_valid):
    X_valid = torch.from_numpy(np.array(X_valid))
    y_valid = torch.from_numpy(np.array(y_valid))
    print("X_valid shape is {}, y_valid shape is {}"
          .format(X_valid.shape, y_valid.shape))
    X_valid = X_valid.type(torch.float32)

    y_valid = y_valid.type(torch.float32)
    X_valid = transforms.Normalize([0.4337, 0.2384, 0.3279], [0.4956, 0.4261, 0.4695])(X_valid)
    valid_tensor = data.TensorDataset(X_valid, y_valid)
    valid_tensor = data.DataLoader(valid_tensor, batch_size=6, shuffle=False, num_workers=8)
    return valid_tensor
