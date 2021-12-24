
from copy import deepcopy

from torch import nn
import numpy as np
import torch
from torch.utils.data.dataset import Subset

from utils.model import CIFARResNet
from utils.data import grouping_default, dataset_split_r_random, get_targets_set, get_targets



def init_settings(trainset, client_num, data_num_per_client, r, server_num, max_delay, max_connection) \
    -> 'tuple[list[Subset], np.ndarray, np.ndarray]':
    """
    return initial
    d: 1*c, datasets on clients
    D: c*s, delay matrix
    B: 1*s, bandwidth vector
    """

    indexes_list = dataset_split_r_random(trainset, client_num, data_num_per_client, r)
    d = [ Subset(trainset, indexes) for indexes in indexes_list ]

    D = np.random.rand(client_num, server_num) * max_delay

    B = np.random.rand(server_num)
    sum = np.sum(B)
    B = B / sum * max_connection
    B = B.astype(int)

    return d, D, B

def init_models(client_num, device) -> 'tuple[nn.Module, list[nn.Module]]':
    """
    return
    model: global model
    models: models on clients
    """
 
    model: nn.Module = CIFARResNet()
    model.to(device)
    # models on all clients
    models: 'list[nn.Module]' = [ model ] * client_num
    for model in models:
        new_model = deepcopy(model.state_dict())
        model.load_state_dict(new_model)
        # model.to(device)

    return model, models



def group_std(d: 'list[Subset]', G: np.ndarray):
    """
    return 
    std of number of all data in groups
    """
    stds = np.zeros((G.shape[1],))
    label_num = len(get_targets_set(d[0].dataset))
    label_list = np.zeros((G.shape[1], label_num))

    targets = get_targets(d[0].dataset)
    for i, client in enumerate(G):
        for j, to_group in enumerate(client):
            if G[i][j] == 1:
                for index in d[i].indices:
                    label_list[j][targets[index]] += 1

    for i, group in enumerate(label_list):
        total_num = np.sum(group)
        avg = total_num / len(group)
        for j, target in enumerate(group):
            stds[i] += (label_list[i][j] - avg) ** 2

    return stds

def group_selection(models: 'list[nn.Module]', model: nn.Module, d, D, l, B, G, A, M) -> np.ndarray:
    """
    return
    G: c*g, grouping matrix
    A: g*s, group assignment matrix
    """
    gradients = np.zeros((len(models),))
    sd_global = model.state_dict()
    for i, model in enumerate(models):
        sd = model.state_dict()
        for key in sd.keys():
            gradients[i] += abs(sd_global[key] - sd[key])
            

    # filter for l
    for i, group in enumerate(M):
        for j, to_server in enumerate(group):
            if M[i][j] > l:
                A[i][j] = 0
    

def init_group_selection(d, D, l, B, G, A, M) -> np.ndarray:
    """
    return 
    A: g*s, initial group assignment matrix
    """
    # filter for l
    for i, group in enumerate(M):
        for j, to_server in enumerate(group):
            if M[i][j] > l:
                A[i][j] = 0
    


def bootstrap(d: list, D: np.ndarray, l, B: list) -> 'tuple[np.ndarray, np.ndarray]':
    """
    return initial
    G: c*g, grouping matrix
    A: g*s, group assignment matrix
    """

    G, A, M = grouping_default(d, D, B)
    A = init_group_selection(d, D, l, B, G, A, M)

    return G, A

def global_iter(d: list, models: 'list[nn.Module]', G: np.ndarray, A: np.ndarray) \
    -> 'tuple[list[nn.Module], nn.Module]':
    """
    return
    models: 1*c, new models on clients
    model: new global model
    """
    pass

def re_assign(d: list, D: np.ndarray, B: list, models, model)-> 'tuple[np.ndarray, np.ndarray]':
    """
    return the next
    G: c*g
    A: g*s
    """
    pass



