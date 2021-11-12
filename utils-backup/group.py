
from torch import nn

from copy import deepcopy

from tasks import Task

class Server:
    def __init__(self, model: nn.Module) -> None:
        self.model: nn.Module = model
        pass

    def download_model(model: nn.Module):
        pass

    def aggregate_model(groups: 'list[Group]'):
        pass

class Trainer:
    def __init__(self, task: Task) -> None:
        self.task: Task = task

    def download_model(self, model: nn.Module):
        self.task.load_state_dict
        self.model = deepcopy(model)

    def upload_model(self) -> nn.Module:
        return self.model

    def train_model() -> None:
        pass

class Client(Trainer):
    def __init__(self, model: nn.Module) -> None:
        self.model: nn.Module = model

    def download_model(model: nn.Module):
        pass

    def upload_model() -> nn.Module:
        pass

    def train_model() -> None:
        pass

class Group:
    def __init__(self, server: Server, subgroups: 'list[Group]' ) -> None:
        self.single_client = True if server == None else False

        # a virtual or real server
        self.server: Server = server
        self.subgroups: 'list[Group]' = subgroups

    
    def download_model(model: nn.Module):
        pass

    def upload_model():
        pass

    def aggregate_model():
        pass

    def distribute_model():
        pass
