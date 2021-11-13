
# code source:
# FashionMNIST: pytorch tutorial quickstart
# SpeechCommand: pytorch tutorial audio
# TextClassification: pytorch tutorial text

from multiprocessing import set_start_method
from utils.hierarchy import Client, Global, Group

from utils.task import Task, Config, TaskCIFAR
from utils.data import dataset_split_r

import torch
from torch.utils.data import DataLoader


def global_run():
    # set configs
    group_epoch_num = 5
    local_epoch_num = 1

    client_num = 50
    group_size = 10
    group_num = int(client_num/group_size)

    data_num = 256
    batch_size = 64
    lr = 0.01
    r = 5

    # create hierarchical structure
    device = "cuda" if torch.cuda.is_available() else "cpu"
    trainset, testset = TaskCIFAR.load_dataset("./data/")
    subsets = dataset_split_r(trainset, client_num, data_num, r)
    config = Config(local_epoch_num, batch_size, lr, device)

    clients = [ Client(TaskCIFAR(subsets[i], config))
        for i in range(client_num) ]
    counter = 0
    groups = []
    for i in range(group_num):
        group_clients = []
        for j in range(group_size):
            group_clients.append(clients[counter])
        group = Group(group_clients, local_epoch_num)
        groups.append(group)
    global_sys = Global(groups, group_epoch_num)

    
    global_epoch_num = 1000
    _, testset = TaskCIFAR.load_dataset("./data/")

    dataloader: DataLoader = DataLoader(testset, batch_size=256,
        shuffle=False)

    for i in range(global_epoch_num):
        global_sys.round()

        acc, loss = global_sys.test_model(dataloader, device)

        print("Epoch %d: accuracy=%.2f" % (i, acc))


if __name__ == "__main__":
    # prepare for multi-proccessing
    # if get_start_method(False) != "spawn":
    # set_start_method("spawn")

    global_run()

