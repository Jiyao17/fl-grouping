
from copy import deepcopy

from torch import nn
import numpy as np
import torch
from torch.utils.data import dataset
from torch.utils.data.dataset import Subset
from torch.utils.data import DataLoader

from utils.model import CIFARResNet
from utils.data import dataset_split_r_random, get_targets_set, get_targets

class Client:
    def __init__(
        self, model: nn.Module, 
        trainset: Subset, 
        lr: float, 
        device: str="cpu", 
        batch_size: int=0, 
        loss=nn.CrossEntropyLoss()
        ) -> None:

        self.model = model
        self.trainset = trainset
        self.lr = lr
        self.device = device
        self.batch_size = batch_size if batch_size != 0 else len(trainset.indices)
        self.loss = loss

        self.trainloader = DataLoader(self.trainset, self.batch_size, True, drop_last=True)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9, weight_decay=0.0001)


    def train(self):
        self.model.to(self.device)
        self.model.train()

        for (image, label) in self.trainloader:

            y = self.model(image.to(self.device))
            loss = self.loss(y, label.to(self.device))

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()


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

def init_clients(d: 'list[Subset]', lr, device, loss=nn.CrossEntropyLoss()) \
    -> 'tuple[nn.Module, list[Client]]':
    """
    return
    model: global model
    models: models on clients
    """
    clients: 'list[Client]' = []
    client_num = len(d)
    model: nn.Module = CIFARResNet()
    for i in range(client_num):
        new_model = CIFARResNet()
        sd = deepcopy(model.state_dict())
        new_model.load_state_dict(sd)

        batch_size = len(d[i])
        clients.append(Client(new_model, d[i], lr, device, batch_size, loss))

    return model, clients

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

def calc_stds(clients: 'list[Client]', G: np.ndarray) -> np.ndarray:
    """
    return 
    std of number of all data in groups
    """
    d = []
    for client in clients:
        d.append(client.trainset)

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

def calc_dist(model: nn.Module, global_model: nn.Module, device) -> np.ndarray:
    """
    return
    1-norm of gradients
    """
    model.to(device)
    global_model.to(device)
    sd_global = global_model.state_dict()
    sd = model.state_dict()

    dist = 0
    for x, y in zip(sd.values(), sd_global.values()):
        dist += (x - y).abs().sum()

    return dist

def calc_dists_by_group(clients: 'list[Client]', model: nn.Module, G)-> np.ndarray:
    dists_group = np.zeros((G.shape[1]))

    for i, dist in enumerate(G):
        for j, to_server in enumerate(G[i]):
            if G[i][j] == 1:
                dist = calc_dist(clients[i].model, model, clients[i].device)
                dists_group[j] += dist
    
    return dists_group

def calc_group_delay(D, G) -> np.ndarray:
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

def group_selection(model: nn.Module, clients: 'list[Client]', l, B, G, M) -> np.ndarray:
    """
    return
    A: g*s, group assignment matrix
    """

    # filter for l
    # A = filter_delay(M, A, l)

    # rank groups

    dists = calc_dists_by_group(clients, model, G)
    stds = calc_stds(clients, G)

    Q = dists - stds
    seq = [ i for i in range(Q.shape[0])]
    Q_sorted = sorted(zip(Q, seq))
    Q_sorted.reverse()

    # filter for B
    # group size
    G_size = np.zeros((G.shape[1],))
    for i, client in enumerate(G):
        for j, to_group in enumerate(client):
            if G[i][j] == 1:
                G_size[j] += 1

    B_temp = np.zeros((B.shape[0],))
    A = np.zeros(M.shape)
    group_assigned = np.zeros(A.shape[0])
    for (quality, seq) in Q_sorted:
        for j, to_server in enumerate(A[seq]):
            if B_temp[j] + G_size[seq] <= B[j] and M[seq][j] <= l and group_assigned[seq] == 0:
                B_temp[j] += G_size[seq]
                A[seq][j] = 1
                group_assigned[seq] = 1
            else:
                A[seq][j] = 0

    return A

def group_aggregation(clients: 'list[Client]', group: 'list[int]'):

    # get clients size
    C_size = []
    for index in group:
        size = len(clients[index].trainset.indices)
        C_size.append(size)
    data_num_sum = sum(C_size)

    # get state dicts
    state_dicts = []
    for index in group:
        client = clients[index]
        state_dicts.append(client.model.state_dict())

    # calculate average model
    state_dict_avg = deepcopy(state_dicts[0]) 
    for key in state_dict_avg.keys():
        state_dict_avg[key] = 0 # state_dict_avg[key] * -1
    for key in state_dict_avg.keys():
        for i in range(len(state_dicts)):
            state_dict_avg[key] += state_dicts[i][key] * (C_size[i] / data_num_sum)
        # state_dict_avg[key] = torch.div(state_dict_avg[key], len(state_dicts))
    
    # update all clients in this group
    selected_clients: list[Client] = [clients[i] for i in group]
    for i, client in enumerate(selected_clients):
        new_sd = deepcopy(state_dicts[i])
        client.model.load_state_dict(new_sd)

def single_group_train(clients: 'list[Client]', group: 'list[int]', group_epoch_num: int):

    for i in range(group_epoch_num):
        # group_single_train(models, group)
        for client_index in group:
            client = clients[client_index]
            model = client.model
            trainset = client.trainset
            device = client.device
            loss_fn = client.loss

            trainloader = DataLoader(trainset, len(trainset.indices), True, drop_last=True)
            model.to(device)
            model.train()

            for (image, label) in trainloader:

                y = model(image.to(device))
                loss = loss_fn(y, label.to(device))
                optimizer = torch.optim.SGD(model.parameters(), lr=client.lr, momentum=0.9, weight_decay=0.0001)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
    
    group_aggregation(clients, group)

def get_selected_groups(clients: 'list[Client]', G: np.ndarray, A: np.ndarray) -> 'tuple[list[list[int]], list[int]]':
    selected_flags: np.ndarray = np.max(A, axis=1)
    selected_groups: list[list[int]] = []
    G_size: list[int] = [ 0 for i in range(A.shape[0])]
    G_T = G.transpose()
    for i, selected in enumerate(selected_flags):
        if selected == 1:
            new_group = []
            for j, client in enumerate(G_T[i]):
                if client == 1:
                    new_group.append(j)
                    G_size[i] += len(clients[j].trainset.indices)

            selected_groups.append(new_group)
    
    return selected_groups, G_size

def global_aggregate(model: nn.Module, clients: 'list[Client]', G, A) -> nn.Module:
    selected_groups, groups_size = get_selected_groups(clients, G, A)
    
    # G_size = get_groups_size(clients, G)
    data_num_sum = sum(groups_size)

    state_dicts = []
    for group in enumerate(selected_groups):
        client = clients[group[0]]
        model = client.model.to(client.device)
        state_dicts.append(model.state_dict())

    # calculate average model
    state_dict_avg = deepcopy(state_dicts[0]) 
    for key in state_dict_avg.keys():
        state_dict_avg[key] = 0 # state_dict_avg[key] * -1

    for key in state_dict_avg.keys():
        for i in range(len(state_dicts)):
            state_dict_avg[key] += state_dicts[i][key] * (groups_size[i] / data_num_sum)
        # state_dict_avg[key] = torch.div(state_dict_avg[key], len(state_dicts))
    
    model.load_state_dict(state_dict_avg)
    return model

def global_distribute(model: nn.Module, clients: 'list[Client]') -> None:
    for client in clients:
        new_sd = deepcopy(model.state_dict())
        client.model.load_state_dict(new_sd)

def print_params(model: nn.Module, num: int):
    counter = 1
    for name, param in model.state_dict().items():
        if counter > num:
            break
        else:
            print(param[0][0], end="")
            counter += 1
    
    print("")

def compare_models(model: nn.Module, clients: 'list[Client]', num: int):

    print_params(model, num)
    print_params(clients[0].model, num)
    print_params(clients[len(clients)//2].model, num)
    print_params(clients[-1].model, num)

def global_train(model: nn.Module, clients: 'list[Client]',  G: np.ndarray, A: np.ndarray, group_epoch_num: int) \
    -> 'tuple[nn.Module, list[nn.Module]]':
    """
    return
    model: new global model
    """
    selected_groups, G_size = get_selected_groups(clients, G, A)
    print("Number of data on selected clients: %d" % (sum(G_size),))

    global_distribute(model, clients)

    # compare_models(model, clients, 1)
    
    for i, group in enumerate(selected_groups):
        single_group_train(clients, group, group_epoch_num)

    # compare_models(model, clients, 1)

    model = global_aggregate(model, clients, G, A)

    # compare_models(model, clients, 1)



    return model





