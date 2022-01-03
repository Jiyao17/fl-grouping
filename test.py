
import numpy as np
import torch
import random


from utils.fed import Config

if __name__ == "__main__":
    # make results reproducible
    np.random.seed(1)
    random.seed(1)

    test_config = Config(
        client_num = 2500, learning_rate=0.1,
        data_num_per_client = 20, local_batch_size = 20,
        group_epoch_num=5,
        r = 5,
        server_num = 10,
        l = 60,
        max_delay = 120,
        max_connection = 500,
        group_selection_interval = 10,
        log_interval=5,
    )

    config = test_config



