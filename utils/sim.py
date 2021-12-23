
from copy import deepcopy

from torch import nn
import numpy as np
import torch
from torch.utils.data.dataset import Subset

from utils.model import CIFARResNet
from utils.data import load_dataset, dataset_split_r_random, grouping



def init_settings(trainset, client_num, data_num_per_client, r, server_num, max_delay, max_connection) \
    -> 'tuple[list, np.ndarray, np.ndarray]':
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


def group_selection(models, model,) -> np.ndarray:
    """
    return
    G: c*g, grouping matrix
    
    """

def bootstrap(d: list, D: np.ndarray, B: list) -> 'tuple[np.ndarray, np.ndarray]':
    """
    return initial
    G: c*g, grouping matrix
    A: g*s, group assignment matrix
    """

    G = grouping(d, D, B)
    pass

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



