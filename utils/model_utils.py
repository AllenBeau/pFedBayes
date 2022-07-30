import json
import os
import torch

IMAGE_SIZE = 28
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE
NUM_CHANNELS = 1

IMAGE_SIZE_CIFAR = 32
NUM_CHANNELS_CIFAR = 3


def read_data(dataset, subset='data'):
    '''parses data in given train and test data directories
    assumes:
    - the data in the input directories are .json files with 
        keys 'users' and 'user_data'
    - the set of train set users is the same as the set of test set users

    Return:
        clients: list of client ids
        groups: list of group ids; empty list if none found
        train_data: dictionary of train data
        test_data: dictionary of test data
    '''

    train_data_dir = os.path.join('data', dataset, subset, 'train')
    test_data_dir = os.path.join('data', dataset, subset, 'test')
    clients = []
    groups = []
    train_data = {}
    test_data = {}

    train_files = os.listdir(train_data_dir)
    train_files = [f for f in train_files if f.endswith('.json')]
    for f in train_files:
        file_path = os.path.join(train_data_dir, f)
        with open(file_path, 'r') as inf:
            cdata = json.load(inf)
        clients.extend(cdata['users'])
        if 'hierarchies' in cdata:
            groups.extend(cdata['hierarchies'])
        train_data.update(cdata['user_data'])

    test_files = os.listdir(test_data_dir)
    test_files = [f for f in test_files if f.endswith('.json')]
    for f in test_files:
        file_path = os.path.join(test_data_dir, f)
        with open(file_path, 'r') as inf:
            cdata = json.load(inf)
        test_data.update(cdata['user_data'])

    clients = list(sorted(train_data.keys()))
    print('train data dist:\n', [len(x['y']) for x in train_data.values()])
    return clients, groups, train_data, test_data


def read_user_data(index, data, dataset, device=None):
    id = data[0][index]
    train_data = data[2][id]
    test_data = data[3][id]
    device = torch.device('cpu') if device is None else device
    X_train, y_train, X_test, y_test = train_data['x'], train_data['y'], test_data['x'], test_data['y']
    if dataset == "Mnist" or dataset == "FMnist":
        X_train, y_train, X_test, y_test = train_data['x'], train_data['y'], test_data['x'], test_data['y']
        X_train = torch.Tensor(X_train).view(-1, NUM_CHANNELS, IMAGE_SIZE, IMAGE_SIZE).type(torch.float32).to(device)
        y_train = torch.Tensor(y_train).type(torch.int64).to(device)
        X_test = torch.Tensor(X_test).view(-1, NUM_CHANNELS, IMAGE_SIZE, IMAGE_SIZE).type(torch.float32).to(device)
        y_test = torch.Tensor(y_test).type(torch.int64).to(device)
    else:
        X_train = torch.Tensor(X_train).type(torch.float32).to(device)
        y_train = torch.Tensor(y_train).type(torch.int64).to(device)
        X_test = torch.Tensor(X_test).type(torch.float32).to(device)
        y_test = torch.Tensor(y_test).type(torch.int64).to(device)

    train_data = [(x, y) for x, y in zip(X_train, y_train)]
    test_data = [(x, y) for x, y in zip(X_test, y_test)]
    return id, train_data, test_data
