
from torch.utils.data.dataset import Dataset


class TrainerConfig:
    def __init__(self,
        dataset: Dataset,
        epoch_num: int,
        batch_size: int,
        lr: float,
    ) -> None:
        self.dataset = dataset
        self.epoch_num = epoch_num
        self.batch_size = batch_size
        self.lr = lr

