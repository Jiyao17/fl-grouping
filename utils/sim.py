
from copy import deepcopy

from torch import nn
import numpy as np
import math
import torch
from torch.utils.data import dataset
from torch.utils.data.dataset import Subset
from torch.utils.data import DataLoader



# def compare_models(model: nn.Module, clients: 'list[Client]', num: int):
#     def print_params(model: nn.Module, num: int):
#         counter = 1
#         for name, param in model.state_dict().items():
#             if counter > num:
#                 break
#             else:
#                 print(param[0][0], end="")
#                 counter += 1
        
#         print("")


#     print_params(model, num)
#     print_params(clients[0].model, num)
#     print_params(clients[len(clients)//2].model, num)
#     print_params(clients[-1].model, num)


# def regroup(G: np.ndarray, A: np.ndarray, s: int) -> 'tuple[np.ndarry, np.ndarry]':
#     "s: each new group contains s old groups"
#     group_num: int = G.shape[1]
#     new_group_size: int = math.ceil(group_num / s)

#     A_T = A.transpose()

#     group2server: list[int] = []
    
#     # get new groups as list
#     new_groups: 'list[list[int]]' = []
#     for i, server in enumerate(A_T):
#         new_group: 'list[int]' = []
#         for j, group in server:
#             if A_T[i][j] == 1:
#                 if len(new_group) < new_group_size:
#                     new_group.append(j)
#                 else:
#                     new_groups.append(new_group)
#                     new_group = []

#     # construct new A
#     new_A = np.zeros((len(new_groups), A.shape[1],))
#     for i, new_group in enumerate(new_groups):
#         one_group = new_group[0]
#         belong_to_server = 0
#         for j, to_server in enumerate(A[one_group]):
#             if to_server == 1:
#                 belong_to_server = j
#                 break
        
#         new_A[i][belong_to_server] = 1
        
#     # construct new G
#     new_G = np.zeros((G.shape[0], len(new_groups),))
#     for i, new_group in enumerate(new_groups):
#         for old_group in new_group:
#             G_T = G.transpose()
#             for k, contain_client in enumerate(G_T[old_group]):
#                 if contain_client == 1:
#                     new_G[k][i] = 1

#     return new_G, new_A
