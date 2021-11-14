
# code source:
# FashionMNIST: pytorch tutorial quickstart
# SpeechCommand: pytorch tutorial audio
# TextClassification: pytorch tutorial text

from copy import deepcopy
from multiprocessing import set_start_method
from utils.hierarchy import Client, Global, Group

from utils.task import TrainConfig, TaskCIFAR
from utils.data import dataset_split_r

import torch
from torch.utils.data import DataLoader

from utils.task import TrainConfig, ExpConfig
from utils.simulator import Simulator


def global_run():

    # create hierarchical structure
    trainset, testset = TaskCIFAR.load_dataset("./data/")
    subsets = dataset_split_r(trainset, client_num, data_num, r)
    config = TrainConfig(local_epoch_num, batch_size, lr, device)

    clients = [ Client(TaskCIFAR(subsets[i], config))
        for i in range(client_num) ]
    counter = 0
    groups = []
    for i in range(group_num):
        group_clients = []
        for j in range(group_size):
            group_clients.append(clients[counter])
            counter += 1
        group = Group(group_clients, local_epoch_num)
        groups.append(group)
    global_sys = Global(groups, group_epoch_num)

    
    _, testset = TaskCIFAR.load_dataset("./data/")

    dataloader: DataLoader = DataLoader(testset, batch_size=256,
        shuffle=False)

    for i in range(global_epoch_num):
        global_sys.round()

        acc, loss = global_sys.test_model(dataloader, device)

        print("Epoch %d: accuracy=%.2f" % (i, acc))


def group_run():
    # create hierarchical structure
    device = "cuda" if torch.cuda.is_available() else "cpu"
    trainset, testset = TaskCIFAR.load_dataset("./data/")
    subsets = dataset_split_r(trainset, client_num, data_num, r)
    config = TrainConfig(local_epoch_num, batch_size, lr, device)

    clients = [ Client(TaskCIFAR(subsets[i], config))
        for i in range(client_num) ]
    counter = 0
    group_clients: list[Client] = []
    for j in range(group_size):
        group_clients.append(clients[counter])
        counter += 1
    group = Group(group_clients, local_epoch_num)

    dataloader: DataLoader = DataLoader(testset, batch_size=256,
        shuffle=False)

    file = open("")
    for i in range(group_epoch_num):
        sd = deepcopy(group.model.state_dict())
        group.distribute_model()
        group.train_model()
        group.aggregate_model()
        sd1 = group.model.state_dict()

        acc, loss = TaskCIFAR.test_model(group_clients[0].get_model(), dataloader, device)

        print("Epoch %d: accuracy=%.2f" % (i, acc))


if __name__ == "__main__":
    # prepare for multi-proccessing
    # if get_start_method(False) != "spawn":
    # set_start_method("spawn")

    # global_run()
    # group_run()

    # default config for iid
    exp_config = ExpConfig("iid", group_epoch_num=500, local_epoch_num=5, result_dir="./cifar/iid/")
    # exp_config = ExpConfig()
    sim = Simulator(exp_config)
