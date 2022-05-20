
import torch

import numpy as np
import random

from utils.fed import GFL, Config
from utils.data import TaskName

if __name__ == "__main__":
    # make results reproducible
    # seed = 5456
    # random.seed(seed)
    # np.random.seed(seed)
    # torch.manual_seed(seed)

    fedavg = Config(
        task_name=TaskName.CIFAR,
        server_num=1, client_num=1000, data_num_range=(10, 50), alpha=0.1,
        sampling_frac=0.2,
        global_epoch_num=500, group_epoch_num=1, local_epoch_num=5,
        lr=0.1, lr_interval=250, local_batch_size=10,
        log_interval=5, 
        # alpha=0.1: sigma = 
        grouping_mode=Config.GroupingMode.NONE, max_group_cv=1000, min_group_size=1,
        # partition_mode=Config.PartitionMode.IID,
        selection_mode=Config.SelectionMode.RANDOM,
        device="cuda",
        data_path="./data/", 
        result_dir="./exp_data/fedavg/",
        test_id=0,
        comment="fed avg, no grouping, random selection",
    )


    cv_grouping_cv_select = Config(
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
        result_dir="./exp_data/grouping/cvg_cvs/",
        test_id=0,
        comment="cvg cvs",
    )

    cv_grouping_rand_select = Config(
        task_name=TaskName.CIFAR,
        server_num=10, client_num=1000, data_num_range=(10, 50), alpha=0.1,
        sampling_frac=0.2,
        global_epoch_num=500, group_epoch_num=5, local_epoch_num=1,
        lr=0.1, lr_interval=250, local_batch_size=10,
        log_interval=5, 
        # alpha=0.1: sigma = 
        grouping_mode=Config.GroupingMode.CV_GREEDY, max_group_cv=0.1, min_group_size=10,
        # partition_mode=Config.PartitionMode.IID,
        selection_mode=Config.SelectionMode.RANDOM,
        device="cuda",
        data_path="./data/", 
        result_dir="./exp_data/grouping/cvg_rs/",
        test_id=0,
        comment="",
    )

    rand_grouping_cv_select = Config(
        task_name=TaskName.CIFAR,
        server_num=10, client_num=1000, data_num_range=(10, 50), alpha=0.1,
        sampling_frac=0.2,
        global_epoch_num=500, group_epoch_num=5, local_epoch_num=1,
        lr=0.1, lr_interval=250, local_batch_size=10,
        log_interval=5, 
        # alpha=0.1: sigma = 
        grouping_mode=Config.GroupingMode.RANDOM, max_group_cv=0.1, min_group_size=10,
        # partition_mode=Config.PartitionMode.IID,
        selection_mode=Config.SelectionMode.PROB_CV,
        device="cuda",
        data_path="./data/", 
        result_dir="./exp_data/rg_cvs/",
        test_id=0,
        comment="",
    )

    rand_grouping_rand_select = Config(
        task_name=TaskName.CIFAR,
        server_num=10, client_num=1000, data_num_range=(10, 50), alpha=0.1,
        sampling_frac=0.2,
        global_epoch_num=500, group_epoch_num=5, local_epoch_num=1,
        lr=0.1, lr_interval=250, local_batch_size=10,
        log_interval=5, 
        # alpha=0.1: sigma = 
        grouping_mode=Config.GroupingMode.RANDOM, max_group_cv=0.1, min_group_size=10,
        # partition_mode=Config.PartitionMode.IID,
        selection_mode=Config.SelectionMode.RANDOM,
        device="cuda",
        data_path="./data/", 
        result_dir="./exp_data/grouping/rg_rs/",
        test_id=0,
        comment="",
    )

    debug = Config(
        task_name=TaskName.CIFAR,
        server_num=5, client_num=100, data_num_range=(10, 50), alpha=0.1,
        sampling_frac=0.2,
        global_epoch_num=500, group_epoch_num=5, local_epoch_num=1,
        lr=0.01, lr_interval=100, local_batch_size=10,
        log_interval=5, 
        # alpha=0.1: sigma = 
        grouping_mode=Config.GroupingMode.CV_GREEDY, max_group_cv=0.1, min_group_size=5,
        # partition_mode=Config.PartitionMode.IID,
        selection_mode=Config.SelectionMode.PROB_CV,
        device="cuda",
        data_path="./data/", 
        result_dir="./exp_data/debug/",
        test_id=0,
        comment="",
    )


def rg_rs_min_group_sizes():
    config = rand_grouping_rand_select

    min_group_sizes = [5, 10, 15, 20, 25]
    for min_group_size in min_group_sizes:
        config.min_group_size = min_group_size
        config.test_id = min_group_size
        config.comment = "cvg cvs " + str(min_group_size)

        gfl = GFL(config)
        gfl.run()




