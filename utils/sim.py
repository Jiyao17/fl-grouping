
from copy import deepcopy

from torch import nn
import numpy as np
import torch
from torch.utils.data.dataset import Subset

from utils.model import CIFARResNet
from utils.data import dataset_split_r_random, get_targets_set, get_targets



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

def find_next(cur_set: set, subunions: 'list[set]') -> int:
    """
    return the index if can add more categories

    return -1 if cannot add more set
    """
    max_len = 0
    pos = -1
    for i, subunion in enumerate(subunions):
        cur_len = len(subunion.difference(cur_set))
        if cur_len > max_len:
            pos = i
            max_len = cur_len
    
    return pos

def clustering(d: 'list[Subset]', group: 'list[int]') -> 'list[list[int]]':
    # sets = datasets_to_target_sets(subsets)
    sets: list[set] = []
    targets = get_targets(d[0].dataset)
    for client in group:
        new_set = set()
        for index in d[client].indices:
            new_set.add(targets[index])
        sets.append(new_set)

    groups: 'list[list[int]]' = []
    # indicates unions of current groups
    unions: 'list[set[int]]' = []
    # group_labels: set = set()
    targets_set = set(targets)
    while len(sets) > 0:
        # try to get a new group
        new_group: 'list[list[int]]' = []
        new_set = set()

        pos = find_next(new_set, sets)
        while pos != -1:
            new_group.append(group[pos])
            new_set = new_set.union(sets[pos])
            group.pop(pos)
            sets.pop(pos)

            pos = find_next(new_set, sets)

        groups.append(new_group)
        unions.append(new_set)
    # replace index with subset
    # for group in groups:
    #     for i in range(len(group)):
    #         group[i] = subsets[group[i]]
    return groups

def grouping_default(d: 'list[Subset]', D) \
    -> 'tuple[np.ndarray, np.ndarray, np.ndarray]':
    """
    Assign each client to its nearest server
    Clustering clients connected to the same server
    l, B not considered yet

    return group and accesory information
    G: grouping matrix
    A: initial assignment matrix, delay and bandwidth not considered
    M: g*s, delay, groups to servers
    """

    # group clients by delay to servers
    groups: list[list[int]] = [ [] for i in range(D.shape[1]) ]
    # clients to their nearest servers
    server_indices = np.argmin(D, 1)
    for i, server in enumerate(server_indices):
        groups[server].append(i)

    group_num = 0
    clusters_list = []
    # cluster clients to the same server
    for group in groups:
        clusters = clustering(d, group)
        clusters_list.append(clusters)
        group_num += len(clusters)

    # get initial G A M without sufficing l & B
    G = np.zeros((len(d), group_num), int)
    A = np.zeros((group_num, D.shape[1]), int)
    # G_size = np.zeros((group_num,))
    # M = -1 * np.ones((group_num, D.shape[1]))

    group_counter = 0
    for server, clusters in enumerate(clusters_list):
        for cluster in clusters:
            max_delay = -1
            for client in cluster:
                G[client][group_counter] = 1

                # group delay
                # delay = D[client][server]
                # if delay > max_delay:
                #     max_delay = delay

            A[group_counter][server] = 1
            # G_size[group_counter] = len(cluster)
            # M[group_counter][server] = max_delay
            group_counter += 1
    
    M = calc_group_delay(D, G)

    return G, M

def calc_stds(d: 'list[Subset]', G: np.ndarray) -> np.ndarray:
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

def calc_dists(models: 'list[nn.Module]', model: nn.Module) -> np.ndarray:
    """
    return
    1-norm of gradients
    """
    dists = np.zeros((len(models),))
    sd_global = model.state_dict()
    for i, model in enumerate(models):
        sd = model.state_dict()
        for x, y in zip(sd.values(), sd_global.values()):
            dists[i] += (x - y).abs().sum()

    return dists

def calc_dists_by_group(models: 'list[nn.Module]', model: nn.Module, G)-> np.ndarray:
    dists = calc_dists(models, model)

    dists_group = np.zeros((G.shape[1]))

    for i, dist in enumerate(dists):
        for j, to_server in enumerate(G[i]):
            if G[i][j] == 1:
                dists_group[j] += dist
    
    return dists_group

def filter_delay(M, A, l):
    for i, group in enumerate(M):
        for j, to_server in enumerate(group):
            if M[i][j] > l:
                A[i][j] = 0
    
    return A

def calc_group_delay(D, G):
    M = np.zeros((G.shape[1], D.shape[1]))

    # get clients in each group
    grouping = [ [] for i in range(G.shape[1])]
    for i, client in enumerate(G):
        for j, to_group in enumerate(client):
            if G[i][j] == 1:
                grouping[j].append(i)

    for i, group in enumerate(M):
        for j, to_server in enumerate(group):
            # get group i delay to server j
            max_delay = 0
            for client in grouping[i]:
                if D[client][j] > max_delay:
                    max_delay = D[client][j]
            M[i][j] = max_delay

    return M

# def rank_groups()

def group_selection(models: 'list[nn.Module]', model: nn.Module, d, l, B, G, M) -> np.ndarray:
    """
    return
    A: g*s, group assignment matrix
    """

    # filter for l
    # A = filter_delay(M, A, l)

    # rank groups
    dists = calc_dists_by_group(models, model, G)
    stds = calc_stds(d, G)

    Q = dists - stds
    rank = [ i for i in range(Q.shape[0])]
    Q_sorted = sorted(zip(Q, rank)).reverse()

    # filter for B
    # group size
    G_size = np.zeros((G.shape[1],))
    for i, client in enumerate(G):
        for j, to_group in enumerate(client):
            if G[i][j] == 1:
                G_size[j] += 1

    B_temp = np.zeros((B.shape[0],))
    A = np.zeros(M.shape)
    for i, group in enumerate(Q_sorted):
        for j, to_server in enumerate(A[i]):
            if B_temp[j] + G_size[i] <= B[j] and M[i][j] <= l:
                B_temp[j] += G_size[i]
                A[i][j] = 1
            else:
                A[i][j] = 0

    return A

    
# def init_group_selection(d, D, l, B, G, A, M) -> np.ndarray:
#     """
#     return 
#     A: g*s, initial group assignment matrix
#     """
#     # filter for l
#     for i, group in enumerate(M):
#         for j, to_server in enumerate(group):
#             if M[i][j] > l:
#                 A[i][j] = 0

def bootstrap(d: list, D: np.ndarray, l, B: list) -> 'tuple[np.ndarray, np.ndarray]':
    """
    return initial
    G: c*g, grouping matrix
    M: g*s, group delay matrix
    """

    G, M = grouping_default(d, D)

    return G, M

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



