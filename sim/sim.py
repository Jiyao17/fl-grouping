
from typing import List

from torch import nn
import numpy as np
from torch.utils.data.dataset import Subset

from utils import CIFARResNet
from utils import load_dataset, dataset_split_r_random

matrix = List[List]

# the global model
m: nn.Module = CIFARResNet()
# models on all clients
models: 'list[nn.Module]' = []

# grouping matrix
G: matrix = [[]]
# group assignment matrix
A: matrix = [[]]

def init_settings(client_num, data_num_per_client, r, server_num, max_delay, max_connection) -> 'tuple[list, matrix, list]':
    """
    init d, D, B
    """

    # datasets on clients
    trainset, _ = load_dataset("../data/", "train")
    indexes_list = dataset_split_r_random(trainset, client_num, data_num_per_client, r)
    d = [ Subset(trainset, indexes) for indexes in indexes_list ]

    # delay matrix
    D = np.random.rand(client_num, server_num) * max_delay

    # bandwidth vector
    B = np.random.rand(server_num)
    sum = np.sum(B)
    B = B / sum * max_connection
    B = B.astype(int)

    return d, D, B

def bootstrap(d: list, D: matrix, B: list) -> 'tuple[matrix, matrix]':
    """
    return: initial G, A
    """
    pass

def global_iter(d: list, models: 'list[nn.Module]', G: matrix, A: matrix) \
    -> 'tuple[list[nn.Module], nn.Module]':
    """
    return: new models on clients, new global model
    """
    pass

def re_assign(d: list, D: matrix, B: list, models, model)-> 'tuple[matrix, matrix]':
    """
    return new G, A
    """
    pass

if __name__ == "__main__":
    # make results reproducible
    np.random.seed(1)

    client_num = 2500
    data_num_per_client = 20
    r = 5
    server_num = 10
    max_delay = 60
    max_connection = 500

    d, D, B = init_settings(client_num, data_num_per_client, r, server_num, max_delay, max_connection)
    


    print(B)
