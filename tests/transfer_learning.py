

from utils.model import *
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
import torch.nn as nn
import torchvision.transforms as tvtf

trainset = CIFAR10(root="data", train=True, download=True, transform=tvtf.ToTensor())
trainloader = DataLoader(trainset, batch_size=1024, shuffle=True)
net = CIFARResNet()
optimizer = torch.optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
loss = nn.CrossEntropyLoss()

# print(net.state_dict().keys())
print("before training=====================")
print(net.convs[8].conv2.weight[-1, -1, -1])
print(net.fc.bias)

for X, y in trainloader:
    y_hat = net(X)
    l = loss(y_hat, y)
    optimizer.zero_grad()
    l.backward()
    optimizer.step()

    break

print("after training=====================")
print(net.convs[8].conv2.weight[-1, -1, -1])
print(net.fc.bias)

print("freeze fc===========================")
net.fc.requires_grad_(False)
params = filter(lambda p: p.requires_grad, net.parameters())
optimizer = torch.optim.SGD(params, lr=0.1, momentum=0.9, weight_decay=5e-4)


for X, y in trainloader:
    y_hat = net(X)
    l = loss(y_hat, y)
    optimizer.zero_grad()
    l.backward()
    optimizer.step()

    break

print("after training=====================")
print(net.convs[8].conv2.weight[-1, -1, -1])
print(net.fc.bias)

print("unfreeze fc===========================")
net.fc.requires_grad_(True)
optimizer = torch.optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)

for X, y in trainloader:
    y_hat = net(X)
    l = loss(y_hat, y)
    optimizer.zero_grad()
    l.backward()
    optimizer.step()

    break

print("after training=====================")
print(net.convs[8].conv2.weight[-1, -1, -1])
print(net.fc.bias)
