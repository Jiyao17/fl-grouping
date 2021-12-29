
import numpy as np
import torch
from torch.utils.data import DataLoader
import random

from utils.sim import group_selection, init_clients, init_settings, bootstrap, global_train
from utils.model import test_model
from utils.data import load_dataset


if __name__ == "__main__":
    # make results reproducible
    np.random.seed(1)
    random.seed(1)

    # optimization settings
    client_num = 2500
    data_num_per_client = 20
    r = 5
    server_num = 10
    l = 60
    max_delay = 90
    max_connection = 500

    # federated learning settings
    data_path = "../data/"
    global_epoch_num = 300
    group_epoch_num = 5
    learning_rate = 0.1
    local_batch_size = data_num_per_client
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # results
    log_interval = 1
    result_file_accu = "./cifar/grouping/accu"
    result_file_loss = "./cifar/grouping/loss"

    # initialize result file IO wrappers
    faccu = open(result_file_accu, "a")
    floss = open(result_file_loss, "a")


    trainset, testset = load_dataset(data_path, "both")
    d, D, B = init_settings(trainset, client_num, data_num_per_client, r, server_num, max_delay, max_connection)
    model, clients = init_clients(d, learning_rate, device)
    testloader = DataLoader(testset, 500, drop_last=True)

    G, M = bootstrap(d, D, l, B)
    
    for i in range(global_epoch_num):
        A = group_selection(model, clients, l, B, G, M)

        model = global_train(model, clients, G, A, group_epoch_num)
        # G, A = re_assign(d, D, B, models, model)

        if (i + 1) % log_interval == 0:
            accu, loss = test_model(model, testloader, device)
            faccu.write("{:.5f} ".format(accu))
            faccu.flush()
            floss.write("{:.5f} ".format(loss))
            floss.flush()

