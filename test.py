
import numpy as np
import torch
import random


from utils.fed import GFL, GFLConfig

if __name__ == "__main__":
    # make results reproducible
    np.random.seed(1)
    random.seed(1)

    torch.cuda.empty_cache()

    folders = ["test", "temp", "grouping"]
    folder = folders[2]

    test_config = GFLConfig(
        client_num = 100, learning_rate=0.1,
        data_num_per_client = 500, local_batch_size = 50,
        global_epoch_num= 300, reselect_interval=300,

        group_epoch_num=5, r = 5, server_num = 1,
        l = 60, max_delay = 60, max_connection = 500,
        log_interval=10,

        min_group_size=1,
        comment="central", 

        result_file_accu="./cifar/" + folder + "/accu",
        result_file_loss="./cifar/" + folder + "/loss",
    )

    config = test_config

    gfl = GFL(config)
    gfl.train()


