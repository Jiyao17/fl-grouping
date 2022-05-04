
from copy import deepcopy
import numpy as np
import torch
import random

from utils.fed import GFL, GFLConfig

if __name__ == "__main__":
    # for AI report: start from line 318


    # make results reproducible
    np.random.seed(1)
    random.seed(1)

    torch.cuda.empty_cache()

    folders = ["central", "fedavg",
        "grouping", "multiserver_grouping", "noniid", "multiserver_noniid"]
    folder = folders[5]

    test_config = GFLConfig(
        client_num = 100, lr=0.1, lr_interval=3000,
        data_num_per_client = 500, local_batch_size = 50,
        global_epoch_num= 300, reselect_interval=300,

        server_num = 1,
        group_epoch_num=5, r = 5, 
        l = 60, max_delay = 60, max_connection = 5000, 
        log_interval=10,

        regroup_size=1, # 1: no regrouping,
        group_size=1, # 1: no grouping,
        comment="multi server noniid", 

        result_file_accu="./cifar/multiserver_fedavg/accu",
        result_file_loss="./cifar/multiserver_fedavg/loss",
    )

    fedavg = GFLConfig(
        client_num = 500, lr=0.5, lr_interval=50,
        data_num_per_client=50, local_batch_size = 50,
        global_epoch_num=100, reselect_interval=1000,

        server_num = 1,
        group_epoch_num=1, local_epoch_num=5,
        r = 2, partition_mode=GFLConfig.PartitionMode.IID_AND_NON_IID, iid_proportion=0.5,
        l = 60, max_delay = 60, max_connection = 5000,
        log_interval=1,

        selection_mode=GFLConfig.SelectionMode.RANDOM,
        grouping_mode=GFLConfig.GroupingMode.IID,
        regroup_size=1, # 1: no regrouping,
        group_size=1, # 1: no grouping,
        comment="full participation",  

        result_file_accu="./cifar/fedavg/accu0",
        result_file_loss="./cifar/fedavg/loss0",
    )

    ms_fedavg = GFLConfig(
        client_num = 1000, lr=0.1, lr_interval=100,
        data_num_per_client = 50, local_batch_size = 50,
        global_epoch_num= 300, reselect_interval=300,

        server_num = 10,
        group_epoch_num=1, local_epoch_num=5, r = 10, 
        l = 60, max_delay = 120, max_connection = 200, 
        log_interval=10,

        regroup_size=1, # 1: no regrouping,
        group_size=1, # 1: no grouping,
        comment="multi server fedavg", 

        result_file_accu="./cifar/multiserver_fedavg/accu",
        result_file_loss="./cifar/multiserver_fedavg/loss",
    )

    noniid = GFLConfig(
        client_num = 20, lr=0.5, lr_interval=50,
        data_num_per_client=1000, local_batch_size = 50,
        global_epoch_num=100, reselect_interval=1,

        server_num = 1,
        group_epoch_num=1, local_epoch_num=5,
        r = 5, partition_mode=GFLConfig.PartitionMode.NONIID, iid_proportion=0,
        l = 60, max_delay = 60, max_connection = 10,
        log_interval=1,

        selection_mode=GFLConfig.SelectionMode.RANDOM,
        grouping_mode=GFLConfig.GroupingMode.IID,
        regroup_size=1, # 1: no regrouping,
        group_size=1, # 1: no grouping,
        comment="single server iid_and_noniid, no grouping, iid selection",

        result_file_accu="./cifar/noniid/accu",
        result_file_loss="./cifar/noniid/loss",
    )

    ms_noniid = GFLConfig(
        client_num = 2500, lr=1, lr_interval=500,
        data_num_per_client=20, local_batch_size = 20,
        global_epoch_num= 1500, reselect_interval=3000,

        server_num = 10,
        group_epoch_num=1, local_epoch_num=1, r = 2, 
        l = 60, max_delay = 60, max_connection = 500, 
        log_interval=50,

        regroup_size=1, # 1: no regrouping,
        group_size=1, # 1: no grouping,
        comment="multi server noniid", 

        result_file_accu="./cifar/multiserver_noniid/accu",
        result_file_loss="./cifar/multiserver_noniid/loss",
    )

    grouping = GFLConfig(
        client_num = 500, lr=0.5, lr_interval=5,
        data_num_per_client = 50, local_batch_size = 50,
        global_epoch_num= 10, reselect_interval=3000,

        server_num=1,
        group_epoch_num=30, local_epoch_num=5, r = 2, 
        l = 60, max_delay = 60, max_connection = 5000, 
        log_interval=1,

        grouping_mode='iid',
        regroup_size=10, # 1: no regrouping,
        group_size=10, # 1: no grouping,
        comment="single server noniid grouping", 

        result_file_accu="./cifar/grouping/accu",
        result_file_loss="./cifar/grouping/loss",
    )

    grouping_noiid = GFLConfig(
        client_num = 100, lr=0.1, lr_interval=5,
        data_num_per_client=100, local_batch_size = 50,
        global_epoch_num=10, reselect_interval=1,

        server_num = 1,
        group_epoch_num=5, local_epoch_num=5, r = 2, 
        l = 60, max_delay = 60, max_connection = 30, 
        log_interval=1,

        grouping_mode='noiid',
        regroup_size=10, # 1: no regrouping,
        group_size=10, # 1: no grouping,
        comment="single server noniid grouping", 

        result_file_accu="./cifar/grouping_noiid/accu0",
        result_file_loss="./cifar/grouping_noiid/loss0",
    )

    grouping_iid = GFLConfig(
        client_num = 500, lr=0.5, lr_interval=7,
        data_num_per_client = 50, local_batch_size = 50,
        global_epoch_num=15, reselect_interval=3000,

        server_num=1,
        group_epoch_num=20, local_epoch_num=5, r = 2, 
        l = 60, max_delay = 60, max_connection = 5000, 
        log_interval=1,

        grouping_mode='iid',
        regroup_size=10, # 1: no regrouping,
        group_size=10, # 1: no grouping,
        comment="single server noniid grouping", 

        result_file_accu="./cifar/grouping_iid/accu0",
        result_file_loss="./cifar/grouping_iid/loss0",
    )

    grouping_random = GFLConfig(
        client_num = 500, lr=0.5, lr_interval=750,
        data_num_per_client = 50, local_batch_size = 50,
        global_epoch_num=1500, reselect_interval=3000,

        server_num = 1,
        group_epoch_num=1, local_epoch_num=5, r = 2, 
        l = 60, max_delay = 60, max_connection = 5000, 
        log_interval=30,

        grouping_mode='random',
        regroup_size=10, # 1: no regrouping,
        group_size=10, # 1: no grouping,
        comment="single server random grouping", 

        result_file_accu="./cifar/grouping_random/accu0",
        result_file_loss="./cifar/grouping_random/loss0",
    )

    ms_grouping = GFLConfig(
        client_num = 2500, lr=1, lr_interval=100,
        data_num_per_client = 20, local_batch_size = 20,
        global_epoch_num= 300, reselect_interval=3000,

        server_num = 10,
        group_epoch_num=5, local_epoch_num=1, r = 2, 
        l = 60, max_delay = 60, max_connection = 500, 
        log_interval=10, 

        regroup_size=20, # 1: no regrouping,
        group_size=10, # 1: no grouping, now allow grouping
        comment="multi server grouping", 

        result_file_accu="./cifar/multiserver_grouping/accu0",
        result_file_loss="./cifar/multiserver_grouping/loss0",
    )

    debug = GFLConfig(
        client_num = 20, lr=0.5, lr_interval=50,
        data_num_per_client=1000, local_batch_size = 50,
        global_epoch_num=100, reselect_interval=1,

        server_num = 1,
        group_epoch_num=1, local_epoch_num=5,
        r = 10, partition_mode=GFLConfig.PartitionMode.IID, iid_proportion=1,
        l = 60, max_delay = 60, max_connection = 10,
        log_interval=1,

        selection_mode=GFLConfig.SelectionMode.RANDOM,
        grouping_mode=GFLConfig.GroupingMode.IID,
        regroup_size=1, # 1: no regrouping,
        group_size=1, # 1: no grouping,
        comment="single server iid_and_noniid no grouping, random selection",


        result_file_accu="./cifar/debug/accu",
        result_file_loss="./cifar/debug/loss",
    )

    test = GFLConfig(
        client_num = 500, lr=1, lr_interval=500,
        data_num_per_client=50, local_batch_size = 50,
        global_epoch_num=100, reselect_interval=1000,

        server_num = 1,
        group_epoch_num=1, local_epoch_num=5,
        r = 2, partition_mode=GFLConfig.PartitionMode.IID_AND_NON_IID, iid_proportion=0.5,
        l = 60, max_delay = 60, max_connection = 100,
        log_interval=1,

        selection_mode=GFLConfig.SelectionMode.STD,
        grouping_mode=GFLConfig.GroupingMode.IID,
        regroup_size=1, # 1: no regrouping,
        group_size=1, # 1: no grouping,
        comment="longer reselect interval",

        result_file_accu="./cifar/test/accu0",
        result_file_loss="./cifar/test/loss0",
    )

    iid = GFLConfig(
        client_num = 20, lr=0.5, lr_interval=50,
        data_num_per_client=1000, local_batch_size = 50,
        global_epoch_num=100, reselect_interval=1,

        server_num = 1,
        group_epoch_num=1, local_epoch_num=5,
        r = 10, partition_mode=GFLConfig.PartitionMode.IID, iid_proportion=1,
        l = 60, max_delay = 60, max_connection = 10,
        log_interval=1,

        selection_mode=GFLConfig.SelectionMode.RANDOM,
        grouping_mode=GFLConfig.GroupingMode.IID,
        regroup_size=1, # 1: no regrouping,
        group_size=1, # 1: no grouping,
        comment="single server iid_and_noniid, no grouping, iid selection",
        result_file_accu="./cifar/iid/accu0",
        result_file_loss="./cifar/iid/loss0",
    )

    selection = GFLConfig(
        client_num = 500, lr=1, lr_interval=50,
        data_num_per_client=50, local_batch_size = 50,
        global_epoch_num=100, reselect_interval=1,

        server_num = 1,
        group_epoch_num=1, local_epoch_num=5,
        r = 2, partition_mode=GFLConfig.PartitionMode.IID_AND_NON_IID, iid_proportion=0.5,
        l = 60, max_delay = 60, max_connection = 100,
        log_interval=1,

        selection_mode=GFLConfig.SelectionMode.STD,
        grouping_mode=GFLConfig.GroupingMode.IID,
        regroup_size=1, # 1: no regrouping,
        group_size=1, # 1: no grouping,
        comment="longer reselect interval",

        result_file_accu="./cifar/selection/low_std_rand_accu0",
        result_file_loss="./cifar/selection/low_std_rand_loss0",
    )

    selection_multi_interval = GFLConfig(
        client_num = 500, lr=5, lr_interval=50,
        data_num_per_client=50, local_batch_size = 50,
        global_epoch_num=100, reselect_interval=1,

        server_num = 1,
        group_epoch_num=1, local_epoch_num=5,
        r = 2, partition_mode=GFLConfig.PartitionMode.IID_AND_NON_IID, iid_proportion=0.5,
        l = 60, max_delay = 60, max_connection = 100,
        log_interval=1,

        selection_mode=GFLConfig.SelectionMode.LOW_STD_RANDOM,
        grouping_mode=GFLConfig.GroupingMode.IID,
        regroup_size=1, # 1: no regrouping,
        group_size=1, # 1: no grouping,
        comment="single server iid_and_noniid no grouping, random selection",

        result_file_accu="./cifar/selection/multi_inter_accu0",
        result_file_loss="./cifar/selection/multi_inter_loss0",
    )

    # start settings for AI report

    ai_report_fed_avg = GFLConfig(
        client_num = 100, lr=0.5, lr_interval=50,
        data_num_per_client=50, local_batch_size = 50,
        global_epoch_num=100, reselect_interval=10000,

        server_num = 1,
        group_epoch_num=1, local_epoch_num=5,
        r = 2, partition_mode=GFLConfig.PartitionMode.NONIID, iid_proportion=0.5,
        l = 60, max_delay = 60, max_connection = 5000,
        log_interval=1,

        selection_mode=GFLConfig.SelectionMode.RANDOM,
        grouping_mode=GFLConfig.GroupingMode.IID,
        regroup_size=1, # 1: no regrouping,
        group_size=1, # 1: no grouping,
        comment="full participation fed avg for ai report",  

        result_file_accu="./cifar/fedavg/accu0",
        result_file_loss="./cifar/fedavg/loss0",
    )

    ai_report_grouping = GFLConfig(
        client_num = 100, lr=0.5, lr_interval=50,
        data_num_per_client=50, local_batch_size = 50,
        global_epoch_num=100, reselect_interval=10000,

        server_num = 1,
        group_epoch_num=5, local_epoch_num=1,
        r = 2, partition_mode=GFLConfig.PartitionMode.NONIID, iid_proportion=0.5,
        l = 60, max_delay = 60, max_connection = 5000,
        log_interval=1,

        selection_mode=GFLConfig.SelectionMode.RANDOM,
        grouping_mode=GFLConfig.GroupingMode.IID,
        regroup_size=10, # 1: no regrouping,
        group_size=10, # 1: no grouping,
        comment="full participation fed avg for ai report",  

        result_file_accu="./cifar/grouping/accu0",
        result_file_loss="./cifar/grouping/loss0",
    )

    # end settings for AI report

    # set config to the settings you want
    # config = ai_report_fed_avg
    config = ai_report_grouping





    exp_num = 1
    # config.use_file(1)
    # gfl = GFL(config)d
    # gfl.train()

    for i in range(exp_num):
        config.use_file(i)
        gfl = GFL(config)
        gfl.train()


