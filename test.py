
import numpy as np
import torch
from torch.utils.data import DataLoader
import random

from utils.sim import group_selection, init_clients, init_settings, grouping_default, global_train
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
        device = "cpu",
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

    # optimization settings
    client_num = 5000
    data_num_per_client = 10
    r = 5
    server_num = 10
    l = 60
    max_delay = 90
    max_connection = 1000
    group_selection_interval = 10

    # federated learning settings
    data_path = "../data/"
    global_epoch_num = 500
    group_epoch_num = 5
    learning_rate = 0.1
    local_batch_size = data_num_per_client
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # results
    log_interval = 5
    result_file_accu = "./cifar/grouping/accu"
    result_file_loss = "./cifar/grouping/loss"

    # initialize result file IO wrappers
    faccu = open(result_file_accu, "a")
    floss = open(result_file_loss, "a")
    faccu.write("\n")
    floss.write("\n")



    trainset, testset = load_dataset(data_path, "both")
    d, D, B = init_settings(trainset, client_num, data_num_per_client, r, server_num, max_delay, max_connection)
    model, clients = init_clients(d, learning_rate, device)
    testloader = DataLoader(testset, 500, drop_last=True)

    G, M = grouping_default(d, D)
    A = group_selection(model, clients, l, B, G, M)
    
    for i in range(global_epoch_num):

        model = global_train(model, clients, G, A, group_epoch_num)
        # G, A = re_assign(d, D, B, models, model)


        if (i + 1) % log_interval == 0:
            accu, loss = test_model(model, testloader, device)
            faccu.write("{:.5f} ".format(accu))
            faccu.flush()
            floss.write("{:.5f} " .format(loss))
            floss.flush()

            print("accuracy and loss at round %d: %.5f, %.5f" % (i, accu, loss))

