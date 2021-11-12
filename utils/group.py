
from typing import overload
from torch import nn

from copy import deepcopy

from torch.utils import data
from torch.utils.data.dataset import Dataset


from tasks import Task


class Server:
    def __init__(self, model: nn.Module) -> None:
        self.model: nn.Module = model
        pass

    def download_model(model: nn.Module):
        pass

    def aggregate_model(groups: 'list[Group]'):
        pass


from trainer import TrainerConfig
class Client():
    def __init__(self, task: Task, config: TrainerConfig) -> None:
        self.config = deepcopy(config)
        self.task: Task = task

    def download_model(self, model: nn.Module):
        self.task.load_model(model)

    def upload_model(self) -> nn.Module:
        return self.task.get_model()

    def train_model(self) -> None:
        self.task.train_model()


class Group():
    def __init__(self, server: Server, clients: 'list[Client]', config: TrainerConfig) -> None:
        super().__init__(config)
        # a virtual or real server
        self.server: Server = server
        self.clients: 'list[Client]' = clients

    def download_model(model: nn.Module):
        pass

    def upload_model():
        pass

    def train_model():
        pass

    def aggregate_model():
        pass

    def distribute_model():
        pass
