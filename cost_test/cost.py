

import time

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

import numpy as np
import scipy.spatial.distance as distance
import matplotlib.pyplot as plt

from utils import CIFARResNet
from utils import load_dataset_CIFAR

def flatten_model(model):
    params = []
    for i, param in enumerate(model0.parameters()):
        params.append(param.detach().cpu().numpy().flatten())

    params = np.concatenate(params)

    return params


trainset, testset = load_dataset_CIFAR("../data/", "both")
model0 = CIFARResNet()
model1 = CIFARResNet()
device = 'cuda'
loss_fn=nn.CrossEntropyLoss()
lr = 0.1

optimizer = torch.optim.SGD(model0.parameters(), lr=lr, momentum=0.9, weight_decay=0.0001) #

dataset_sizes = list(range(4, 50 + 1))
group_sizes = list(range(5, 50 + 1))

# distance overhead
def distance_overhead(model0, model1, group_sizes):
    dist_overhead = []
    for size in group_sizes:
        time_start = time.time()

        for i in range(size):
            params0 = flatten_model(model0)
            for j in range(size - 1):
                params1 = flatten_model(model1)
                cd = distance.cosine(params0, params1)
                ud = distance.euclidean(params0, params1)

        time_end = time.time()
        real_time = (time_end - time_start) / 2
        dist_overhead.append(real_time)
        # print("size: {}, time: {}".format(size, real_time))

    return dist_overhead


def training_overhead(model0: CIFARResNet, dataset_sizes):
    train_overhead = []
    for size in dataset_sizes:
        real_trainset = Subset(trainset, list(range(size)))
        trainloader = DataLoader(real_trainset, 50, True) #, drop_last=True


        model0.to(device)
        model0.train()

        # local epoch = 1
        time_start = time.time()
        for (image, label) in trainloader:
            y = model0(image.to(device))
            loss = loss_fn(y, label.to(device))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        time_end = time.time()

        real_time = (time_end - time_start) * 5
        train_overhead.append(real_time)
        # print("size: {}, time: {}".format(size, real_time))

    return train_overhead
        

dist_overhead = distance_overhead(model0, model1, group_sizes)
train_overhead = training_overhead(model0, dataset_sizes)
dataset_sizes = dataset_sizes[1:]
train_overhead = train_overhead[1:]

plt.plot(group_sizes, dist_overhead, label='distance overhead')
plt.plot(dataset_sizes, train_overhead, label='training overhead')
plt.legend()
plt.savefig('./dist_train_overhead.png')

coefs_dist = np.polyfit(group_sizes, dist_overhead, 2)
coefs_train = np.polyfit(dataset_sizes, train_overhead, 1)
print("distance coefficiences: {}".format(coefs_dist))
print("training coefficiences: {}".format(coefs_train))
# distance coefficiences: [ 0.00072459 -0.00023117 -0.00164357]
# training coefficiences: [0.00294225 0.0653882 ]





