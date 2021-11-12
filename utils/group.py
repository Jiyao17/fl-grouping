
from typing import overload
from torch import nn

from copy import deepcopy

from torch.utils import data
from torch.utils.data.dataset import Dataset


from utils.task import Task, TaskConfig


class Server:
    def __init__(self, task: Task) -> None:
        self.task: Task = task
        pass

    def download_model(model: nn.Module):
        pass

    def aggregate_model(clients: 'list[Client]'):
        pass


class Client():
    def __init__(self, task: Task, config: TaskConfig) -> None:
        self.config = deepcopy(config)
        self.task: Task = task

    def set_model(self, model: nn.Module):
        self.task.set_model(model)

    def get_model(self) -> nn.Module:
        return self.task.get_model()

    def train_model(self) -> None:
        self.task.train_model()


class Group():
    def __init__(self, clients: 'list[Client]', config: TaskConfig) -> None:
        self.clients: 'list[Client]' = clients
        self.config = config

        self.model: nn.Module = deepcopy(clients[0].task.get_model())

    def download_model(self, model: nn.Module):
        self.model = deepcopy(model)

    def upload_model(self):
        return self.model

    def train_model(self):
        for client in self.clients:
            for i in range(self.config.epoch_num):
                client.train_model()

    def aggregate_model(self):
        state_dicts = [
            client.get_model().state_dict()
            for client in self.clients
            ]

        weights = [ len(client.task.dataset) for client in self.clients ]
        weights_sum = sum(weights)

        # calculate average model
        state_dict_avg = deepcopy(state_dicts[0]) 
        for key in state_dict_avg.keys():
            state_dict_avg[key] = 0 # state_dict_avg[key] * -1

        for key in state_dict_avg.keys():
            for i in range(len(state_dicts)):
                state_dict_avg[key] += state_dicts[i][key] * (weights[i] / weights_sum)
            # state_dict_avg[key] = torch.div(state_dict_avg[key], len(state_dicts))
        
        self.model.load_state_dict(state_dict_avg)

    def distribute_model(self):
        for client in self.clients:
            client.set_model(self.model)


class Global():
    def __init__(self) -> None:
        pass