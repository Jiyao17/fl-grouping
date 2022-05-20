
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
        result_file_accu="./exp_data/fedavg/accu", 
        result_file_loss="./exp_data/fedavg/loss",
        result_file_pic="./exp_data/fedavg/pic.png",
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
        result_file_accu="./exp_data/grouping/accu_cvg_cvs", 
        result_file_loss="./exp_data/grouping/loss_cvg_cvs",
        result_file_pic="./exp_data/grouping/pic_cvg_cvs.png",
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
        result_file_accu="./exp_data/grouping/accu_cvg_rs", 
        result_file_loss="./exp_data/grouping/loss_cvg_rs",
        result_file_pic="./exp_data/grouping/pic_cvg_rs.png",
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
        result_file_accu="./exp_data/grouping/accu_rg_cvs", 
        result_file_loss="./exp_data/grouping/loss_rg_cvs",
        result_file_pic="./exp_data/grouping/pic_rg_cvs.png",
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
        result_file_accu="./exp_data/grouping/accu_rg_rs", 
        result_file_loss="./exp_data/grouping/loss_rg_rs",
        result_file_pic="./exp_data/grouping/pic_rg_rs.png",
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
        result_file_accu="./exp_data/debug/accu", 
        result_file_loss="./exp_data/debug/loss",
        result_file_pic="./exp_data/debug/pic",
        comment="",
    )



    config = cv_grouping_cv_select

    exp_num = 3


    gfl = GFL(config)
    gfl.run()




