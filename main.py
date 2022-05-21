
import torch

import numpy as np
import random

from utils.fed import GFL, Config
from utils.data import TaskName

fedavg = Config(
    task_name=TaskName.CIFAR,
    server_num=1, client_num=1000, data_num_range=(10, 50), alpha=0.1,
    sampling_frac=0.2,
    global_epoch_num=500, group_epoch_num=1, local_epoch_num=5,
    lr=0.1, lr_interval=250, local_batch_size=50,
    log_interval=5, 
    # alpha=0.1: sigma = 
    grouping_mode=Config.GroupingMode.NONE, max_group_cv=1000, min_group_size=1,
    # partition_mode=Config.PartitionMode.IID,
    selection_mode=Config.SelectionMode.RANDOM,
    device="cuda",
    data_path="./data/", 
    result_dir="./exp_data/fedavg/",
    test_mark="",
    comment="fed avg, no grouping, random selection",
)

cv_grouping_cv_select = Config(
    task_name=TaskName.CIFAR,
    server_num=10, client_num=1000, data_num_range=(10, 50), alpha=0.1,
    sampling_frac=0.2,
    global_epoch_num=200, group_epoch_num=5, local_epoch_num=1,
    lr=0.5, lr_interval=100, local_batch_size=50,
    log_interval=5, 
    # alpha=0.1: sigma = 
    grouping_mode=Config.GroupingMode.CV_GREEDY, max_group_cv=0.1, min_group_size=10,
    # partition_mode=Config.PartitionMode.IID,
    selection_mode=Config.SelectionMode.PROB_CV,
    device="cuda",
    data_path="./data/", 
    result_dir="./exp_data/grouping/cvg_cvs/",
    test_mark="0",
    comment="cvg cvs",
)

cv_grouping_rand_select = Config(
    task_name=TaskName.CIFAR,
    server_num=5, client_num=1000, data_num_range=(10, 50), alpha=0.1,
    sampling_frac=0.2,
    global_epoch_num=200, group_epoch_num=5, local_epoch_num=1,
    lr=0.5, lr_interval=100, local_batch_size=50,
    log_interval=5, 
    # alpha=0.1: sigma = 
    grouping_mode=Config.GroupingMode.CV_GREEDY, max_group_cv=0.1, min_group_size=10,
    # partition_mode=Config.PartitionMode.IID,
    selection_mode=Config.SelectionMode.RANDOM,
    device="cuda",
    data_path="./data/", 
    result_dir="./exp_data/grouping/cvg_rs/",
    test_mark="0",
    comment="",
)

rand_grouping_cv_select = Config(
    task_name=TaskName.CIFAR,
    server_num=5, client_num=1000, data_num_range=(10, 50), alpha=0.1,
    sampling_frac=0.2,
    global_epoch_num=200, group_epoch_num=5, local_epoch_num=1,
    lr=0.5, lr_interval=100, local_batch_size=50,
    log_interval=5, 
    # alpha=0.1: sigma = 
    grouping_mode=Config.GroupingMode.RANDOM, max_group_cv=0.1, min_group_size=10,
    # partition_mode=Config.PartitionMode.IID,
    selection_mode=Config.SelectionMode.PROB_CV,
    device="cuda",
    data_path="./data/", 
    result_dir="./exp_data/rg_cvs/",
    test_mark="",
    comment="",
)

rand_grouping_rand_select = Config(
    task_name=TaskName.CIFAR,
    server_num=10, client_num=1000, data_num_range=(10, 50), alpha=0.1,
    sampling_frac=0.2,
    global_epoch_num=200, group_epoch_num=5, local_epoch_num=1,
    lr=0.5, lr_interval=100, local_batch_size=50,
    log_interval=5, 
    # alpha=0.1: sigma = 
    grouping_mode=Config.GroupingMode.RANDOM, max_group_cv=0.1, min_group_size=10,
    # partition_mode=Config.PartitionMode.IID,
    selection_mode=Config.SelectionMode.RANDOM,
    device="cuda",
    data_path="./data/", 
    result_dir="./exp_data/grouping/rg_rs/",
    test_mark="",
    comment="",
)

debug = Config(
    task_name=TaskName.CIFAR,
    server_num=5, client_num=1000, data_num_range=(10, 50), alpha=0.1,
    sampling_frac=0.2,
    global_epoch_num=500, group_epoch_num=5, local_epoch_num=1,
    lr=0.1, lr_interval=250, local_batch_size=50,
    log_interval=5, 
    # alpha=0.1: sigma = 
    grouping_mode=Config.GroupingMode.CV_GREEDY, max_group_cv=0.1, min_group_size=10,
    # partition_mode=Config.PartitionMode.IID,
    selection_mode=Config.SelectionMode.PROB_CV,
    device="cuda",
    data_path="./data/", 
    result_dir="./exp_data/debug/",
    test_mark="",
    comment="",
)


def rg_rs_min_group_sizes():
    config = rand_grouping_rand_select

    min_group_sizes = [5, 10, 15, 20, 25]
    # min_group_sizes = [5, 10, 15, 20]
    for min_group_size in min_group_sizes:
        config.min_group_size = min_group_size
        config.test_mark = "_gs" + str(min_group_size)

        gfl = GFL(config)
        gfl.run()

def fedavg_ng_rs():
    config = fedavg

    gfl = GFL(config)
    gfl.run()

def grouping_cvg_cvs(mark: str):
    config = cv_grouping_cv_select
    config.test_mark = mark

    gfl = GFL(config)
    gfl.run()

def debug_test(mark: str):
    config = debug
    config.test_mark = mark

    gfl = GFL(config)
    gfl.run()

if __name__ == "__main__":

    seed = None
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
    # fedavg_ng_rs()
    # grouping_cvg_cvs("")
    rg_rs_min_group_sizes()
    # debug_test("no_greedy_first")



