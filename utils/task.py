
from copy import deepcopy
from typing import overload
from torch import nn, optim
import torch

import torchvision
import torchvision.transforms as tvtf

from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader, dataloader

from utils.models import CIFAR_CNN



class ExpConfig:
    TEST_TYPE = ("grouping", "noniid", "test")
    TASK_NAME = ("CIFAR", )

    def __init__(self,
        test_type: str="grouping",
        task_name: str="CIFAR",
        global_epoch_num: int=500,
        group_epoch_num: int=5,
        local_epoch_num: int=1,
        client_num: int=100,
        group_size: int=10,
        group_num: int=10,
        local_data_num: int=500,
        batch_size: int=32,
        lr: int=0.001,
        noniid_degree: float=5,
        device: int="cuda",
        datapath: int="./data/",
        result_dir: str="./cifar/noniid/",
        simulation_num: int=3,
        simulation_index: int=0,
        ) -> None:
        self.task_type: str = test_type
        self.task_name: str = task_name
        self.global_epoch_num: int = global_epoch_num
        self.group_epoch_num: int = group_epoch_num
        self.local_epoch_num: int = local_epoch_num
        self.client_num: int = client_num
        self.group_size: int = group_size
        self.group_num: int = group_num
        self.local_data_num: int = local_data_num
        self.batch_size: int = batch_size
        self.lr: int = lr
        self.noniid_degree: float = noniid_degree
        self.datapath: int = datapath
        self.device: int = device
        self.result_dir: str = result_dir
        self.simulation_num: int = simulation_num
        self.simulation_index: int = simulation_index


    def get_class(self):
        if self.task_name == ExpConfig.TASK_NAME[0]:
            return TaskCIFAR
        else:
            raise "Unspported task"


class Task:
    @overload
    @staticmethod
    def load_dataset(data_path: str):
        pass
    
    @overload
    @staticmethod
    def test_model(model: nn.Module, testloader: DataLoader, device: str) \
        -> 'tuple[float, float]':
        pass

    # create new model while calling __init__ in subclasses
    def __init__(self, model: nn.Module, trainset: Dataset, config: ExpConfig) -> None:
        self.model: nn.Module = model
        self.trainset = trainset
        self.config = config

    def set_model(self, model: nn.Module):
        # self.optmizaer
        self.model.load_state_dict(deepcopy(model.state_dict()))

    def get_model(self) -> nn.Module:
        return self.model

    @overload
    def train_model(self):
        pass

    @overload
    def test_model():
        pass


class TaskCIFAR(Task):
    loss = nn.CrossEntropyLoss()

    classes = ('plane', 'car', 'bird', 'cat', 'deer',
        'dog', 'frog', 'horse', 'ship', 'truck')

    @staticmethod
    def load_dataset(data_path: str, type: str="both"):
        transform = tvtf.Compose(
            [tvtf.ToTensor(),
                tvtf.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        trainset, testset = None, None
        if type != "test":
            trainset = torchvision.datasets.CIFAR10(root=data_path, train=True,
                download=True, transform=transform)
        if type != "train":
            testset = torchvision.datasets.CIFAR10(root=data_path, train=False,
                download=True, transform=transform)

        return (trainset, testset)

    @staticmethod
    def test_model(model: nn.Module, testloader: DataLoader, device: str) \
        -> 'tuple[float, float]':
        model.to(device)
        model.eval()

        loss = TaskCIFAR.loss
        size = 0
        correct: float = 0.0
        test_loss: float = 0.0
        
        # with torch.no_grad():
        for samples, labels in testloader:
            pred = model(samples.to(device))
            correct += (pred.argmax(1) == labels.to(device)).type(torch.float).sum().item()
            if loss is not None:
                test_loss += loss(pred, labels.to(device)).item()

            size += len(samples)

        correct /= 1.0*size
        test_loss /= 1.0*size
        return correct, test_loss

    def __init__(self, trainset: Dataset, config: ExpConfig) -> None:
        super().__init__(CIFAR_CNN(), trainset, config)

        self.trainloader: DataLoader = DataLoader(self.trainset, batch_size=self.config.batch_size,
            shuffle=True)
        # self.testloader: DataLoader = DataLoader(self.testset, batch_size=512,
        #     shuffle=False)
        self.loss = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.config.lr, momentum=0.9)

    def train_model(self):
        self.model.to(self.config.device)
        self.model.train()

        # running_loss = 0
        for epoch in range(self.config.local_epoch_num):
            for (samples, lables) in self.trainloader:

                y = self.model(samples.to(self.config.device))
                loss = self.loss(y, lables.to(self.config.device))

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # running_loss += loss.item()



# class UniTask:
#     def __init__(self, config: ExpConfig) -> None:
#         self.config = deepcopy(config)
    
#         self.task_class: type = None
#         if self.config.task_name == "CIFAR":
#             self.task_class = TaskCIFAR




