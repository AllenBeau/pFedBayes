from tqdm import trange
import numpy as np
import random
import json
import os
import torch
import torchvision
import torchvision.transforms as transforms

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
trainset = torchvision.datasets.MNIST(root='./', train=True, download=True, transform=transform)
testset = torchvision.datasets.MNIST(root='./', train=False, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=len(trainset.data), shuffle=False)
testloader = torch.utils.data.DataLoader(testset, batch_size=len(testset.data), shuffle=False)

for _, train_data in enumerate(trainloader, 0):
    trainset.data, trainset.targets = train_data
for _, train_data in enumerate(testloader, 0):
    testset.data, testset.targets = train_data

random.seed(1)
np.random.seed(1)
NUM_USERS = 10
NUM_LABELS = 5

# Setup directory for train/test data
train_path = 'data/train/mnist_train.json'
test_path = 'data/test/mnist_test.json'
dir_path = os.path.dirname(train_path)
if not os.path.exists(dir_path):
    os.makedirs(dir_path)
dir_path = os.path.dirname(test_path)
if not os.path.exists(dir_path):
    os.makedirs(dir_path)

fmnist_data_image = []
fmnist_data_label = []

fmnist_data_image.extend(trainset.data.cpu().detach().numpy())
fmnist_data_image.extend(testset.data.cpu().detach().numpy())
fmnist_data_label.extend(trainset.targets.cpu().detach().numpy())
fmnist_data_label.extend(testset.targets.cpu().detach().numpy())
fmnist_data_image = np.array(fmnist_data_image)
fmnist_data_label = np.array(fmnist_data_label)

fmnist_data = []
for i in trange(10):
    idx = fmnist_data_label == i
    fmnist_data.append(fmnist_data_image[idx])

print("\nNumb samples of each label:\n", [len(v) for v in fmnist_data])
users_lables = []

# ASSIGN TAGS TO EACH USER:
for user in trange(NUM_USERS):
    for j in range(NUM_LABELS):  # 4 labels for each users
        l = (user * NUM_LABELS + j) % 10
        users_lables.append(l)
unique, counts = np.unique(users_lables, return_counts=True)
print("--------------")
print(unique, counts)


def ram_dom_gen(total, size):
    print(total)
    temp = []
    for i in range(size - 1):
        val = np.random.randint(total // (size + 1), total // 2)
        temp.append(1000)
        # total -= val
    temp.append(1000)
    print(temp)
    return temp


number_sample = []
for total_value, count in zip(fmnist_data, counts):
    temp = ram_dom_gen(len(total_value), count)
    number_sample.append(temp)
print("--------------")
print(number_sample)

i = 0
number_samples = []
for i in range(len(number_sample[0])):
    for sample in number_sample:
        print(sample)
        number_samples.append(sample[i])

print("--------------")
print(number_samples)

# CREATE USER DATA SPLIT
X = [[] for _ in range(NUM_USERS)]
y = [[] for _ in range(NUM_USERS)]
idx = np.zeros(10, dtype=np.int64)
count = 0
for user in trange(NUM_USERS):
    for j in range(NUM_LABELS):  # 4 labels for each users
        l = (user * NUM_LABELS + j) % 10
        print("value of L", l)
        print("value of count", count)
        num_samples = number_samples[count]  # num sample
        count = count + 1
        if idx[l] + num_samples <= len(fmnist_data[l]):
            X[user] += fmnist_data[l][idx[l]:idx[l] + num_samples].tolist()
            y[user] += (l * np.ones(num_samples)).tolist()
            idx[l] += num_samples
            print("check len os user:", user, j, "len data", len(X[user]), num_samples)

print("IDX:", idx)  # counting samples for each labels

# Create data structure
train_data = {'users': [], 'user_data': {}, 'num_samples': []}
test_data = {'users': [], 'user_data': {}, 'num_samples': []}

for i in range(NUM_USERS):
    uname = 'f_{0:05d}'.format(i)

    combined = list(zip(X[i], y[i]))
    random.shuffle(combined)
    X[i][:], y[i][:] = zip(*combined)
    num_samples = len(X[i])
    train_len = int(0.2 * num_samples)
    test_len = num_samples - train_len

    train_data['users'].append(uname)
    train_data['user_data'][uname] = {'x': X[i][:train_len], 'y': y[i][:train_len]}
    train_data['num_samples'].append(train_len)
    test_data['users'].append(uname)
    test_data['user_data'][uname] = {'x': X[i][train_len:], 'y': y[i][train_len:]}
    test_data['num_samples'].append(test_len)

print("Num_samples:", train_data['num_samples'])
print("Total_samples:", sum(train_data['num_samples'] + test_data['num_samples']))

with open(train_path, 'w') as outfile:
    json.dump(train_data, outfile)
with open(test_path, 'w') as outfile:
    json.dump(test_data, outfile)

print("Finish Generating Samples")
