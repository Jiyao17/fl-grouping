
import numpy as np
import torch
from torch.utils.data import DataLoader

from utils.sim import init_settings, init_models, bootstrap, global_train, re_assign
from utils.model import test_model
from utils.data import load_dataset


if __name__ == "__main__":
    # make results reproducible
    np.random.seed(1)

    # optimization settings
    client_num = 100
    data_num_per_client = 500
    r = 5
    server_num = 10
    l = 60
    max_delay = 90
    max_connection = 30

    # federated learning settings
    data_path = "../data/"
    global_epoch_num = 300
    local_epoch_num = 5
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
    # model, models = init_models(client_num, device)
    testloader = DataLoader(testset, 500, drop_last=True)

    G, A = bootstrap(d, D, l, B)
    
    for i in range(global_epoch_num):
        models, model = global_train(d, models, G, A)
        G, A = re_assign(d, D, B, models, model)

        if i + 1 % log_interval == 0:
            accu, loss = test_model(model, testloader, device)
            faccu.write("{:.5f} ".format(accu))
            faccu.flush()
            floss.write("{:.5f} ".format(loss))
            floss.flush()

