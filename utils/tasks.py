
from torch import nn

import torchvision
import torchvision.transforms as tvtf

class Task:
    def __init__(self, model: nn.Module) -> None:
        self.model: nn.Module = model

    def load_model(self, model: nn.Module):
        pass

    def get_model(self) -> nn.Module:
        pass

    def train_model(self):
        pass


class TaskCIFAR(Task):
    def __init__(self, model: nn.Module) -> None:
        super().__init__(model)

        

