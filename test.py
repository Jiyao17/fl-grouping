
import numpy as np
import torch
from torch.utils.data import DataLoader
import random

from utils.sim import global_train_group_selection, init_clients, init_settings, grouping_default, global_train
from utils.model import test_model
from utils.data import load_dataset

class Config():
    def __init__(self,
        client_num = 5000,
        data_num_per_client = 10,
        r = 5,
        server_num = 10,
        l = 60,
        max_delay = 90,
        max_connection = 1000,
        group_selection_interval = 10,

        # federated learning settings
        data_path = "../data/",
        global_epoch_num = 500,
        group_epoch_num = 5,
        learning_rate = 0.1,
        local_batch_size = 10,
        device = "cuda",
        # results
        log_interval = 5,
        result_file_accu = "./cifar/grouping/accu",
        result_file_loss = "./cifar/grouping/loss",
    ) -> None:

        self.client_num = client_num
        self.data_num_per_client = data_num_per_client
        self.r = r
        self.server_num = server_num
        self.l = l
        self.max_delay = max_delay
        self.max_connection = max_connection
        self.group_selection_interval = group_selection_interval


        # federated learning settings
        self.data_path = data_path
        self.global_epoch_num = global_epoch_num
        self.group_epoch_num = group_epoch_num
        self.learning_rate = learning_rate
        self.local_batch_size = local_batch_size
        self.device = device

        # results
        self.log_interval = log_interval
        self.result_file_accu = result_file_accu
        self.result_file_loss = result_file_loss

if __name__ == "__main__":
    # make results reproducible
    np.random.seed(1)
    random.seed(1)

    test_config = Config(
        client_num = 200, learning_rate=0.1,
        data_num_per_client = 100, local_batch_size = 100,
        group_epoch_num=5,
        r = 3,
        server_num = 10,
        l = 60,
        max_delay = 90,
        max_connection = 200,
        group_selection_interval = 20,
        log_interval=1,
    )

    config = test_config

    faccu = open(config.result_file_accu, "a")
    floss = open(config.result_file_loss, "a")
    faccu.write("\n" + str(vars(config)) + "\n")
    floss.write("\n" + str(vars(config)) + "\n")
    faccu.flush()
    floss.flush()

    trainset, testset = load_dataset(config.data_path, "both")
    d, D, B = init_settings(trainset, config.client_num, config.data_num_per_client, config.r, config.server_num, config.max_delay, config.max_connection)
    
    labels = trainset.targets
    for i in range(len(d)):
        for index in d[i].indices:
            print(labels[index], end="")
        print("")
    print(D)
    print(B)
    model, clients = init_clients(d, config.learning_rate, config.device)
    testloader = DataLoader(testset, 1000, drop_last=True)

    G, M = grouping_default(d, D)

    print(G)
    print(M)
    
    for i in range(config.global_epoch_num):
        # if i % config.group_selection_interval == 0:
        if i == 0:
            model, epoch_loss, A = global_train_group_selection(model, clients, config.l, B, G, M)
        else:
            model, epoch_loss = global_train(model, clients, G, A, config.group_epoch_num)

        if (i + 1) % config.log_interval == 0:
            accu, loss = test_model(model, testloader, config.device)
            faccu.write("{:.5f} ".format(accu))
            faccu.flush()
            floss.write("{:.5f} " .format(loss))
            floss.flush()

            print("accuracy, loss, training loss at round %d: %.5f, %.5f, %.5f" % (i, accu, loss, epoch_loss))

