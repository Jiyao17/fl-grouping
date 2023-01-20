import numpy as np
import os

from enum import Enum
import copy
import random

from utils.data import TaskName, load_dataset, DatasetPartitioner, quick_draw
from utils.model import CIFARResNet, test_model, SpeechCommand
from utils.fed import Client, Config

from torch.utils.data.dataset import Dataset, Subset
from torch.utils.data import DataLoader

from torchaudio.datasets import SPEECHCOMMANDS
from torchaudio.transforms import Resample
import torch.nn.functional as F
from torch import nn, optim, Tensor
import torch

        
base_config = Config(
    task_name=TaskName.CIFAR,
    server_num=3, client_num=300, data_num_range=(20, 201), alpha=(0.1, 0.1),
    sampling_frac=0.2, budget=10**8,
    global_epoch_num=1000, 
    # the following line may vary
    group_epoch_num=10, local_epoch_num=2,
    lr=0.01, lr_interval=1000, local_batch_size=10,
    log_interval=5, 
    # the following two lines may vary
    grouping_mode=Config.GroupingMode.RANDOM, max_group_cv=0.5, min_group_size=10,
    selection_mode=Config.SelectionMode.RANDOM,
    aggregation_option=Config.AggregationOption.WEIGHTED_AVERAGE,
    device="cuda",
    train_method=Config.TrainMethod.SGD,
    data_path="./data/", 
    # the following 2 lines may vary
    result_dir="./exp_data/grouping/rg_rs/", 
    test_mark="",
    comment="",
)

def merge_model_sds(state_dicts: 'list[dict[str, torch.Tensor]]', weights: 'list[float]') -> 'dict[str, torch.Tensor]':
    # num_sds = len(sds)
    # # sd_merged = copy.deepcopy(sds[0])
    # sd_merged = {}
    # for key in sd_merged.keys():
    #     sd_merged[key] = 0
    #     # sd_merged[key] = sd_merged[key].to(sds[i][key].device)
    #     for i in range(num_sds):
    #         sd_merged[key] += sds[i][key]
    #     sd_merged[key] /= num_sds * 1.0

    # return sd_merged
    # weight = 1.0 / len(state_dicts)
    avg_state_dict = copy.deepcopy(state_dicts[0])
    for key in avg_state_dict.keys():
        avg_state_dict[key] = avg_state_dict[key] * weights[0]

    

    for key in avg_state_dict.keys():
        for i in range(1, len(state_dicts)):
            avg_state_dict[key] = avg_state_dict[key].to('cuda')
            state_dicts[i][key] = state_dicts[i][key].to('cuda')

            avg_state_dict[key] += state_dicts[i][key] * weights[i]
    
    return avg_state_dict



trainset, testset = load_dataset(TaskName.CIFAR)
testloader = DataLoader(testset, batch_size=1000)

partitioner = DatasetPartitioner(trainset, 1, (1000, 1001), (100000, 2000000), TaskName.CIFAR)
# test_partitioner = DatasetPartitioner(testset, 1, (100, 100), (100000, 2000000), TaskName.CIFAR)
# label_type_num = partitioner.label_type_num
distributions = partitioner.get_distributions()
subsets_indices = partitioner.get_subsets()
distri1 = [100, 0, 0, 0, 0, 5, 100, 0, 0, 0]
distri2 = [0, 0, 0, 0, 200, 0, 5, 100,  0, 0]
distri3 = [0, 50, 0, 0, 0, 0, 0, 0, 200, 20]
ctrainset = partitioner.generate_new_dataset(distri3)
ctrainloader = DataLoader(ctrainset, batch_size=100, shuffle=True)
ctestset = partitioner.generate_new_dataset(distri3)
ctestloader = DataLoader(ctestset, batch_size=100, shuffle=True)
dtrainset = partitioner.generate_new_dataset(distri1)
dtrainloader = DataLoader(dtrainset, batch_size=100, shuffle=True)
dtestset = partitioner.generate_new_dataset(distri1)
dtestloader = DataLoader(dtestset, batch_size=100, shuffle=True)
etrainset = partitioner.generate_new_dataset(distri2)
etrainloader = DataLoader(etrainset, batch_size=100, shuffle=True)
etestset = partitioner.generate_new_dataset(distri2)
etestloader = DataLoader(etestset, batch_size=100, shuffle=True)


# partitioner.draw(20,"./pic/dubug.png")
# global model, global drift in SCAFFOLD, clients
subsets_indices.append(ctrainset)
subsets_indices.append(dtrainset)
subsets_indices.append(etrainset)
_, _, clients = Client.init_clients(subsets_indices, base_config)
client0 = clients[0]

for i in range(5):
    client0.train()
    accu, _ = test_model(client0.model, testloader, base_config.device)
    print("generalized accu: ", accu)
    accu, _ = test_model(client0.model, ctestloader, base_config.device)
    print("personalized accu: ", accu)
    accu, _ = test_model(client0.model, dtestloader, base_config.device)
    print("personalized accu: ", accu)
    accu, _ = test_model(client0.model, etestloader, base_config.device)
    print("personalized accu: ", accu)


client1 = clients[1]
sd = copy.deepcopy(client0.model.state_dict())
client1.model.load_state_dict(sd)
print("client1 model loaded")
client2 = clients[2]
sd = copy.deepcopy(client0.model.state_dict())
client2.model.load_state_dict(sd)
print("client2 model loaded")

# accu, _ = test_model(client1.model, testloader, base_config.device)
# print(accu)

for i in range(10):
    client1.train()
    client2.train()
    accu, _ = test_model(client1.model, ctestloader, base_config.device)
    print("personalized accu1: ", accu)
    accu, _ = test_model(client1.model, testloader, base_config.device)
    print("generalized accu2: ", accu)
    accu, _ = test_model(client2.model, dtestloader, base_config.device)
    print("personalized accu2: ", accu)
    accu, _ = test_model(client2.model, testloader, base_config.device)
    print("generalized accu2: ", accu)

    sd1 = copy.deepcopy(client1.model.state_dict())
    sd2 = copy.deepcopy(client2.model.state_dict())
    merged_sd = merge_model_sds([sd1, sd2], [0.5, 0.5])
    client0.model.load_state_dict(merged_sd)
    accu, _ = test_model(client0.model, testloader, base_config.device)
    print("merged accu: ", accu)



    

client2 = clients[2]
sd = copy.deepcopy(client0.model.state_dict())
client2.model.load_state_dict(sd)
print("client2 model loaded")
accu, _ = test_model(client2.model, testloader, base_config.device)
print(accu)

for i in range(5):
    client2.train()
    accu, _ = test_model(client2.model, dtestloader, base_config.device)
    print("personalized accu: ", accu)
    accu, _ = test_model(client2.model, testloader, base_config.device)
    print("generalized accu: ", accu)

client3 = clients[3]
sd = copy.deepcopy(client0.model.state_dict())
client3.model.load_state_dict(sd)
print("client3 model loaded")
accu, _ = test_model(client3.model, testloader, base_config.device)
print(accu)

for i in range(5):
    client3.train()
    accu, _ = test_model(client3.model, etestloader, base_config.device)
    print("personalized accu: ", accu)
    accu, _ = test_model(client3.model, testloader, base_config.device)
    print("generalized accu: ", accu)

sds = [client1.model.state_dict(), client2.model.state_dict(), client3.model.state_dict()]
weights = [1/len(sds)] * len(sds)
sd = merge_model_sds(sds, weights)
client1.model.load_state_dict(sd)
print("client1 model loaded")
accu, _ = test_model(client1.model, testloader, base_config.device)
print("generalized accu: ", accu)
