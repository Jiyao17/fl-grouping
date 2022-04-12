
import numpy as np
import torch
import random


from utils.fed import GFL, GFLConfig

if __name__ == "__main__":
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
        client_num = 500, lr=0.5, lr_interval=5,
        data_num_per_client = 50, local_batch_size = 50,
        global_epoch_num= 10, reselect_interval=3000,

        server_num = 1,
        group_epoch_num=1, local_epoch_num=150, r = 2, 
        l = 60, max_delay = 60, max_connection = 5000,
        log_interval=1,

        grouping_mode='no',
        regroup_size=1, # 1: no regrouping,
        group_size=1, # 1: no grouping,
        comment="single server fedavg",

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
        client_num = 2500, lr=1, lr_interval=100,
        data_num_per_client = 20, local_batch_size = 20,
        global_epoch_num= 300, reselect_interval=3000,

        server_num = 10,
        group_epoch_num=1, local_epoch_num=5, r = 2, 
        l = 60, max_delay = 60, max_connection = 500, 
        log_interval=10,

        regroup_size=1, # 1: no regrouping,
        group_size=1, # 1: no grouping,
        comment="single server noniid", 

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
        client_num = 500, lr=0.5, lr_interval=5,
        data_num_per_client=50, local_batch_size = 50,
        global_epoch_num=10, reselect_interval=3000,

        server_num = 1,
        group_epoch_num=30, local_epoch_num=5, r = 2, 
        l = 60, max_delay = 60, max_connection = 5000, 
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
        client_num = 500, lr=1, lr_interval=5,
        data_num_per_client = 50, local_batch_size = 50,
        global_epoch_num= 10, reselect_interval=3000,

        server_num=1,
        group_epoch_num=10, local_epoch_num=10, r = 2, 
        l = 60, max_delay = 60, max_connection = 5000, 
        log_interval=1,

        grouping_mode='iid',
        regroup_size=10, # 1: no regrouping,
        group_size=10, # 1: no grouping,
        comment="single server noniid grouping", 

        result_file_accu="./cifar/debug/accu",
        result_file_loss="./cifar/debug/loss",
    )

    test = GFLConfig(
        client_num = 500, lr=0.1, lr_interval=5,
        data_num_per_client = 50, local_batch_size = 50,
        global_epoch_num= 10, reselect_interval=3000,

        server_num=1,
        group_epoch_num=10, local_epoch_num=10, r = 2, 
        l = 60, max_delay = 60, max_connection = 5000, 
        log_interval=1,

        grouping_mode='iid',
        regroup_size=10, # 1: no regrouping,
        group_size=10, # 1: no grouping,
        comment="single server noniid grouping", 

        result_file_accu="./cifar/test/accu",
        result_file_loss="./cifar/test/loss",
    )

    iid = GFLConfig(
        client_num = 2500, lr=0.1, lr_interval=100,
        data_num_per_client=20, local_batch_size = 20,
        global_epoch_num= 300, reselect_interval=300,

        server_num = 10,
        group_epoch_num=5, local_epoch_num=1, r = 2, 
        l = 60, max_delay = 120, max_connection = 500, 
        log_interval=10,

        group_size=1, # 1: grouping
        regroup_size=1, # 1: regrouping,

        comment="test",

        result_file_accu="./cifar/iid/accu",
        result_file_loss="./cifar/iid/loss",
    )

    config = grouping_random
    exp_num = 3

    # config.use_file(1)
    # gfl = GFL(config)
    # gfl.train()

    for i in range(exp_num):
        config.use_file(i)
        gfl = GFL(config)
        gfl.train()


