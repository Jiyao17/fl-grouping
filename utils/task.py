
from copy import deepcopy
from typing import overload
from torch import nn, optim
import torch

import torchvision
import torchvision.transforms as tvtf

from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader

from utils.models import CIFAR_CNN


class Config:
    def __init__(self,
        # train: bool,
        # dataset: Dataset,
        # global_epoch_num: int,
        # group_epoch_num: int,
        local_epoch_num: int,
        batch_size: int,
        lr: float,
        device: str,
    ) -> None:
        # self.train = train
        # self.dataset = dataset
        # self.global_epoch_num = global_epoch_num
        # self.group_epoch_num = group_epoch_num
        self.local_epoch_num = local_epoch_num
        self.batch_size = batch_size
        self.lr = lr
        self.device = device


class Task:
    def __init__(self, model: nn.Module, dataset: Dataset, config: Config) -> None:
        self.model: nn.Module = deepcopy(model)
        self.dataset = dataset
        self.config = config

    def set_model(self, model: nn.Module):
        self.model.load_state_dict(deepcopy(model.state_dict()))

    def get_model(self) -> nn.Module:
        return self.model

    @overload
    def train_model(self):
        pass
 
    @overload
    def test_model(self):
        pass


class TaskCIFAR(Task):

    classes = ('plane', 'car', 'bird', 'cat', 'deer',
        'dog', 'frog', 'horse', 'ship', 'truck')

    @staticmethod
    def load_dataset(data_path: str):
        transform = tvtf.Compose(
            [tvtf.ToTensor(),
                tvtf.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        trainset = torchvision.datasets.CIFAR10(root=data_path, train=True,
                download=True, transform=transform)
        testset = torchvision.datasets.CIFAR10(root=data_path, train=False,
                download=True, transform=transform)

        return (trainset, testset)

    @staticmethod
    def test_model(model: nn.Module, dataloader: DataLoader, device):
        model.to(device)
        model.eval()

        size = 0
        correct: float = 0.0
        test_loss: float = 0.0
        
        # with torch.no_grad():
        for samples, labels in dataloader:
            pred = model(samples.to(device))
            # test_loss += loss(pred, labels.to(device)).item()
            correct += (pred.argmax(1) == labels.to(device)).type(torch.float).sum().item()
            size += len(samples)
        correct /= 1.0*size
        test_loss /= 1.0*size
        return correct, test_loss

    def __init__(self, dataset: Dataset, config: Config) -> None:
        super().__init__(CIFAR_CNN(), dataset, config)

        self.dataloader: DataLoader = DataLoader(self.dataset, batch_size=self.config.batch_size,
            shuffle=True)
        self.loss = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.config.lr, momentum=0.9)

    def train_model(self):
        self.model.to(self.config.device)
        self.model.train()

        # running_loss = 0
        for epoch in range(self.config.local_epoch_num):
            for (samples, lables) in self.dataloader:

                y = self.model(samples.to(self.config.device))
                loss = self.loss(y, lables.to(self.config.device))

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # running_loss += loss.item()


            





