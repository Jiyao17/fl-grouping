
import torch

import numpy as np
import random

from utils.fed import GFL, Config

if __name__ == "__main__":
    # make results reproducible
    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    config = Config(
        task_name='CIFAR',
        server_num=10, client_num=1000, data_num_range=(10, 50), alpha=0.1,
        sampling_num=100,
        global_epoch_num=500, group_epoch_num=1, local_epoch_num=5,
        lr=0.1, lr_interval=100, local_batch_size=10,
        log_interval=5, 
        # alpha=0.1: sigma = 
        grouping_mode=Config.GroupingMode.CV_GREEDY, cv=0.1, min_group_size=10,
        partition_mode=Config.PartitionMode.IID,
        selection_mode=Config.SelectionMode.GRADIENT_RANKING,
        device="cuda",
        data_path="./data/", 
        result_file_accu="./cifar/grouping/accu", 
        result_file_loss="./cifar/grouping/loss",
        comment="",
    )

    gfl = GFL(config)

    gfl.run()




