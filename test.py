
from tokenize import group
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
        client_num = 1000, lr=0.1, lr_interval=100,
        data_num_per_client = 50, local_batch_size = 50,
        global_epoch_num= 300, reselect_interval=300,

        server_num = 1,
        group_epoch_num=1, local_epoch_num=5, r = 2, 
        l = 60, max_delay = 60, max_connection = 5000, 
        log_interval=5,

        regroup_size=1, # 1: no regrouping,
        group_size=1, # 1: no grouping,
        comment="single server fedavg", 

        result_file_accu="./cifar/fedavg/accu",
        result_file_loss="./cifar/fedavg/loss",
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
        client_num = 2500, lr=1, lr_interval=100,
        data_num_per_client = 20, local_batch_size = 20,
        global_epoch_num= 300, reselect_interval=3000,

        server_num = 10,
        group_epoch_num=5, local_epoch_num=1, r = 2, 
        l = 60, max_delay = 60, max_connection = 500, 
        log_interval=10,

        regroup_size=20, # 1: no regrouping,
        group_size=10, # 1: no grouping,
        comment="single server grouping", 

        result_file_accu="./cifar/grouping/accu",
        result_file_loss="./cifar/grouping/loss",
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

        result_file_accu="./cifar/multiserver_grouping/accu",
        result_file_loss="./cifar/multiserver_grouping/loss",
    )

    debug = GFLConfig(
        client_num = 1000, lr=1, lr_interval=3000,
        data_num_per_client=50, local_batch_size = 50,

        global_epoch_num= 1000, reselect_interval=3000,

        server_num = 1,
        group_epoch_num=1, local_epoch_num=1, r = 2, 
        l = 60, max_delay = 60, max_connection = 200, 
        log_interval=25,

        # max, min = 1, 1: non-iid FedAvg
        # max, min = 10, 1: iid-grouping (group size <= 10), no regroup, GFL
        # max, min = 1, 10: no grouping, regroup to size >= 10
        # max, min = 10, 20: iid-grouping (group size <= 10) and regroup to size >= 20, AnonyGFL
        group_size=1, # 1: grouping
        regroup_size=1, # 1: regrouping,

        comment="grouping, more group epoch",

        result_file_accu="./cifar/debug/accu",
        result_file_loss="./cifar/debug/loss",
    )

    test = GFLConfig(
        client_num = 1000, lr=1, lr_interval=40,
        data_num_per_client=50, local_batch_size = 50,
        global_epoch_num= 200, reselect_interval=300,

        server_num = 1,
        group_epoch_num=1, local_epoch_num=5, r = 2, 
        l = 60, max_delay = 60, max_connection = 200, 
        log_interval=10,

        group_size=1, # 1: grouping
        regroup_size=1, # 1: regrouping,

        comment="test",

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

    config = ms_grouping

    gfl = GFL(config)
    gfl.train()


