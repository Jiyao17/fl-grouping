
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
        client_num = 100, learning_rate=0.1,
        data_num_per_client = 500, local_batch_size = 50,
        global_epoch_num= 300, reselect_interval=300,

        server_num = 1,
        group_epoch_num=5, r = 5, 
        l = 60, max_delay = 60, max_connection = 5000, 
        log_interval=10,

        min_group_size=1, # 1: no regrouping,
        max_group_size=1, # 1: no grouping,
        comment="multi server noniid", 

        result_file_accu="./cifar/multiserver_fedavg/accu",
        result_file_loss="./cifar/multiserver_fedavg/loss",
    )

    fedavg = GFLConfig(
        client_num = 100, learning_rate=0.1,
        data_num_per_client = 50, local_batch_size = 50,
        global_epoch_num= 300, reselect_interval=300,

        server_num = 1,
        group_epoch_num=1, local_epoch_num=5, r = 10, 
        l = 60, max_delay = 60, max_connection = 5000, 
        log_interval=10,

        min_group_size=1, # 1: no regrouping,
        max_group_size=1, # 1: no grouping,
        comment="single server fedavg", 

        result_file_accu="./cifar/fedavg/accu",
        result_file_loss="./cifar/fedavg/loss",
    )


    ms_fedavg = GFLConfig(
        client_num = 100, learning_rate=0.1,
        data_num_per_client = 50, local_batch_size = 50,
        global_epoch_num= 300, reselect_interval=300,

        server_num = 10,
        group_epoch_num=1, local_epoch_num=5, r = 10, 
        l = 60, max_delay = 60, max_connection = 5000, 
        log_interval=10,

        min_group_size=1, # 1: no regrouping,
        max_group_size=1, # 1: no grouping,
        comment="multi server fedavg", 

        result_file_accu="./cifar/multiserver_fedavg/accu",
        result_file_loss="./cifar/multiserver_fedavg/loss",
    )

    noniid = GFLConfig(
        client_num = 100, learning_rate=0.1,
        data_num_per_client = 50, local_batch_size = 50,
        global_epoch_num= 300, reselect_interval=300,

        server_num = 1,
        group_epoch_num=1, local_epoch_num=5, r = 5, 
        l = 60, max_delay = 60, max_connection = 5000, 
        log_interval=10,

        min_group_size=1, # 1: no regrouping,
        max_group_size=1, # 1: no grouping,
        comment="single server noniid", 

        result_file_accu="./cifar/noniid/accu",
        result_file_loss="./cifar/noniid/loss",
    )

    debug = GFLConfig(
        client_num = 100, learning_rate=0.1,
        data_num_per_client = 500, local_batch_size = 50,
        global_epoch_num= 300, reselect_interval=300,

        server_num = 10,
        group_epoch_num=5, r = 3, 
        l = 60, max_delay = 60, max_connection = 5000, 
        log_interval=5,

        # max, min = 1, 1: non-iid FedAvg
        # max, min = 10, 1: iid-grouping (group size <= 10), no regroup, GFL
        # max, min = 1, 10: no grouping, regroup to size >= 10
        # max, min = 10, 20: iid-grouping (group size <= 10) and regroup to size >= 20, AnonyGFL
        max_group_size=1, # 1: grouping
        min_group_size=1, # 1: regrouping,

        comment="multiserver noniid", 

        result_file_accu="./cifar/debug/accu",
        result_file_loss="./cifar/debug/loss",
    )


    config = noniid

    gfl = GFL(config)
    gfl.train()


