
import torch

import numpy as np
import random
import copy
from multiprocessing import Process

from utils.fed import GFL, Config
from utils.data import TaskName

base_config = Config(
    task_name=TaskName.CIFAR,
    server_num=1, client_num=200, data_num_range=(20, 201), alpha=(0.1, 0.1),
    sampling_frac=0.2, budget=10**8,
    global_epoch_num=1000,
    # the following line may vary
    group_epoch_num=10, local_epoch_num=2,
    lr=0.01, lr_interval=1000, local_batch_size=10,
    log_interval=5, 
    # the following two lines may vary
    grouping_mode=Config.GroupingMode.RANDOM, max_group_cv=100.0, min_group_size=5,
    selection_mode=Config.SelectionMode.RANDOM,
    aggregation_option=Config.AggregationOption.WEIGHTED_AVERAGE,
    device="cuda",
    train_method=Config.TrainMethod.SGD,
    data_path="./data/", 
    # the following 2 lines may vary
    result_dir="./exp_data/debug/", 
    test_mark="_distribution",
    comment="",
)

    

if __name__ == "__main__":
    gfl = GFL(base_config)
    gfl.run()





