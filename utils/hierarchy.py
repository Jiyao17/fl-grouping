
from typing import overload
from torch import nn
import torch

from copy import deepcopy
from torch import Tensor
from torch.optim.lr_scheduler import ExponentialLR

from torch.utils import data
from torch.utils.data import DataLoader


from utils.task import Task, ExpConfig, TaskCIFAR


class Client:
    def __init__(self, task: Task, config: ExpConfig) -> None:
        self.task: Task = task
        self.config = config

    def set_model(self, model: nn.Module):
        self.task.set_model(model)

    def get_model(self) -> nn.Module:
        return self.task.get_model()

    def train_model(self) -> None:
        self.task.train_model()


class Group:
    def __init__(self, clients: 'list[Client]', config: ExpConfig, model: nn.Module=None) -> None:
        self.clients: 'list[Client]' = clients
        self.config = config
        self.model: nn.Module = model

        self.weights = [ len(client.task.trainset) for client in self.clients ]
        self.weights_sum = sum(self.weights)

    def set_model(self, model: nn.Module):
        self.model.load_state_dict(deepcopy(model.state_dict()))
        self.model.to(self.config.device)

    def get_model(self):
        return self.model

    def train_model(self):
        for client in self.clients:
            for i in range(self.config.local_epoch_num):
                client.train_model()

    def aggregate_model(self):
        state_dicts = [
            client.get_model().state_dict()
            for client in self.clients
            ]

        # calculate average model
        state_dict_avg = deepcopy(state_dicts[0]) 
        for key in state_dict_avg.keys():
            state_dict_avg[key] = 0 # state_dict_avg[key] * -1

        for key in state_dict_avg.keys():
            for i in range(len(state_dicts)):
                state_dict_avg[key] += state_dicts[i][key] * (self.weights[i] / self.weights_sum)
            # state_dict_avg[key] = torch.div(state_dict_avg[key], len(state_dicts))
        
        self.model.load_state_dict(state_dict_avg)
        self.model.to(self.config.device)

    def distribute_model(self):
        for client in self.clients:
            client.set_model(self.model)

    def round(self):
        self.distribute_model()
        self.train_model()
        self.aggregate_model()


class Global:
    def __init__(self, groups: 'list[Group]', config: ExpConfig, model: nn.Module=None) -> None:
        self.groups: 'list[Group]' = groups
        self.config = config
        self.model: nn.Module = model

        self.weights = [ group.weights_sum for group in self.groups ]
        self.weights_sum = sum(self.weights)

    def set_model(self, model: nn.Module):
        self.model.load_state_dict(deepcopy(model.state_dict()))
        self.model.to(self.config.device)
        
    def get_model(self):
        return self.model

    def train_model(self):
        for group in self.groups:
            for i in range(self.config.group_epoch_num):
                group.round()

    def aggregate_model(self):
        for group in self.groups:
            group.aggregate_model()

        state_dicts = [
            group.get_model().state_dict()
            for group in self.groups
            ]

        # calculate average model
        state_dict_avg = deepcopy(state_dicts[0]) 
        for key in state_dict_avg.keys():
            state_dict_avg[key] = 0 # state_dict_avg[key] * -1

        for key in state_dict_avg.keys():
            for i in range(len(state_dicts)):
                state_dict_avg[key] += state_dicts[i][key] * (self.weights[i] / self.weights_sum)
            # state_dict_avg[key] = torch.div(state_dict_avg[key], len(state_dicts))
        
        self.model.load_state_dict(state_dict_avg)
        self.model.to(self.config.device)

    def distribute_model(self):
        for group in self.groups:
            group.set_model(self.model)
            group.distribute_model()

    def round(self):
        self.distribute_model()
        self.train_model()
        self.aggregate_model()

    # def test_model(self, dataloader: DataLoader, device):
    #     self.model.to(device)
    #     self.model.eval()

    #     size = 0
    #     correct: float = 0.0
    #     test_loss: float = 0.0
        
    #     with torch.no_grad():
    #         for samples, labels in dataloader:
    #             pred = self.model(samples.to(device))
    #             # test_loss += self.loss(pred, labels.to(device)).item()
    #             correct += (pred.argmax(1) == labels.to(device)).type(torch.float).sum().item()
    #             size += len(samples)
    #     correct /= 1.0*size
    #     test_loss /= 1.0*size
    #     return correct, test_loss