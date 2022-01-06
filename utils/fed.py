
from copy import deepcopy

from torch.utils.data import DataLoader
from torch import nn
from torch.utils.data.dataset import Subset
import torch


import numpy as np

from utils.data import load_dataset
from utils.sim import init_settings
from utils.model import test_model, CIFARResNet
from utils.data import dataset_split_r_random, get_targets_set, get_targets


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



class Client:
    @staticmethod
    def init_clients(d: 'list[Subset]', lr, local_epoch_num, device, batch_size: int=0, loss=nn.CrossEntropyLoss()) \
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

            batch_size = len(d[i]) if batch_size == 0 else batch_size
            clients.append(Client(new_model, d[i], lr, local_epoch_num, device, batch_size, loss))

        return model, clients

    def __init__(
        self, model: nn.Module, 
        trainset: Subset, 
        lr: float, 
        local_epoch_num=5,
        device: str="cpu", 
        batch_size: int=0, 
        loss_fn=nn.CrossEntropyLoss()
        ) -> None:

        self.model = model
        self.trainset = trainset
        self.lr = lr
        self.local_epoch_num = local_epoch_num
        self.device = device
        self.batch_size = batch_size if batch_size != 0 else len(trainset.indices)
        self.loss_fn = loss_fn

        self.trainloader = DataLoader(self.trainset, self.batch_size, True, drop_last=True)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9) #, weight_decay=0.0001

        self.train_loss = None

    def train(self):
        self.model.to(self.device)
        self.model.train()

        self.train_loss = 0

        for i in range(self.local_epoch_num):
            for (image, label) in self.trainloader:
                y = self.model(image.to(self.device))
                loss = self.loss_fn(y, label.to(self.device))
                self.train_loss += loss.item()

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        
        return self.train_loss / self.local_epoch_num



class GFLConfig:
    def __init__(self, task_name="CIFAR", client_num=5000, data_num_per_client=10, r=5, server_num=10, 
        l=60, max_delay=90, max_connection=1000,
        data_path="../data/", global_epoch_num=500, group_epoch_num=1,
        local_epoch_num=5, learning_rate=0.1, 
        local_batch_size=10, device="cuda", log_interval=5, 
        result_file_accu="./cifar/grouping/accu", 
        result_file_loss="./cifar/grouping/loss",
        comment="",
        # GFL specific settings
        reselect_interval=20,
        min_group_size=20,
        max_group_size=100,
        ) -> None:

        self.task_name = task_name
        self.client_num = client_num
        self.data_num_per_client = data_num_per_client
        self.r = r
        self.server_num = server_num
        self.l = l
        self.max_delay = max_delay
        self.max_connection = max_connection
        # federated learning settings
        self.data_path = data_path
        self.global_epoch_num = global_epoch_num
        self.group_epoch_num = group_epoch_num
        self.local_epoch_num = local_epoch_num
        self.learning_rate = learning_rate
        self.batch_size = local_batch_size
        self.device = device
        # results
        self.log_interval = log_interval
        self.result_file_accu = result_file_accu
        self.result_file_loss = result_file_loss
        
        self.comment = comment
        
        self.reselect_interval = reselect_interval
        self.min_group_size = min_group_size
        self.max_group_size = max_group_size


class GFL:

    def set_all_groups(self) -> 'tuple[list[list[int]], list[int]]':
        # set the following two fields
        # using self.G
        self.all_groups = []
        self.all_groups_size = []

        G_T = self.G.transpose()
        for i in range(self.G.shape[1]):
            new_group = []
            size = 0
            for j, client in enumerate(G_T[i]):
                if client == 1:
                    new_group.append(j)
                    size += len(self.clients[j].trainset.indices)

            self.all_groups.append(new_group)
            self.all_groups_size.append(size)
        
    def set_selected_groups(self) -> 'tuple[list[list[int]], list[int]]':
        # set the following two fields
        # using self.G, self.A
        self.selected_groups = []
        self.selected_groups_size = []

        selected_flags: np.ndarray = np.max(self.A, axis=1)
        G_T = self.G.transpose()
        for i, selected in enumerate(selected_flags):
            if selected == 1:
                new_group = []
                size = 0
                for j, client in enumerate(G_T[i]):
                    if client == 1:
                        new_group.append(j)
                        size += len(self.clients[j].trainset.indices)

                self.selected_groups.append(new_group)
                self.selected_groups_size.append(size)

    def __init__(self, config: GFLConfig) -> None:
        self.config = deepcopy(config)

        self.faccu = open(self.config.result_file_accu, "a")
        self.floss = open(self.config.result_file_loss, "a")
        self.faccu.write("\nconfig:" + str(vars(self.config)) + "\n")
        self.floss.write("\nconfig:" + str(vars(self.config)) + "\n")
        self.faccu.flush()
        self.floss.flush()

        self.trainset, self.testset = load_dataset(
            self.config.task_name, self.config.data_path)
        self.d, self.D, self.B = init_settings(self.trainset, self.config.client_num,
            self.config.data_num_per_client, self.config.r, self.config.server_num,
            self.config.max_delay, self.config.max_connection)
        self.testloader = DataLoader(self.testset, 1000, drop_last=True)
        self.model, self.clients = Client.init_clients(self.d, self.config.learning_rate, self.config.local_epoch_num, self.config.device, self.config.batch_size)


        self.G: np.ndarray = None
        self.M: np.ndarray = None
        self.grouping_default()
        
        self.selected_groups: list[list[int]] = []
        self.selected_groups_size: list[int] = []
        self.all_groups: list[list[int]] = []
        self.all_groups_size: list[int] = []

        # labels = trainset.targets
        # for i in range(len(d)):
        #     for index in d[i].indices:
        #         print(labels[index], end="")
        #     print("")
        # print(D)
        # print(B)

        # print(G)
        # print(M)

    def train(self):
        for i in range(self.config.global_epoch_num):
            if i % self.config.reselect_interval == 0:
                self.group_selection()
                
            epoch_loss = self.global_train()

            if (i + 1) % self.config.log_interval == 0:
                accu, loss = test_model(self.model, self.testloader, self.config.device)

                self.faccu.write("{:.5f} ".format(accu))
                self.faccu.flush()
                self.floss.write("{:.5f} " .format(loss))
                self.floss.flush()

                print("accuracy, loss, training loss at round %d: %.5f, %.5f, %.5f" % 
                    (i, accu, loss, epoch_loss))

    def grouping_default(self) -> 'tuple[np.ndarray, np.ndarray, np.ndarray]':
        """
        set
        G: grouping matrix
        M: g*s, delay, groups to servers
        """
        
        def clustering(group: 'list[int]') -> 'list[list[int]]':
            """
            group a client with another one as long as the group is not iid
            if the clients are already iid, then all groups have only one client
            """
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

            # sets = datasets_to_target_sets(subsets)
            sets: list[set] = []
            targets = get_targets(self.d[0].dataset)
            for client in group:
                new_set = set()
                for index in self.d[client].indices:
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
                size_counter = 0

                pos = find_next(new_set, sets)
                while pos != -1 and size_counter < self.config.max_group_size:
                    new_group.append(group[pos])
                    new_set = new_set.union(sets[pos])
                    size_counter += 1
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
 
        def regroup(clusters: 'list[list[int]]', cluster_size: int) -> 'list[list[int]]':
            """
            merge any group smaller than given size to another group until reach the size
            reset G
            """
            new_clusters = []

            new_cluster = []
            for cluster in clusters:
                if len(new_cluster) + len(cluster) < cluster_size:
                    new_cluster += cluster
                else:
                    new_cluster += cluster
                    new_clusters.append(new_cluster)
                    new_cluster = []
            if new_cluster != []:
                new_clusters.append(new_cluster)
            
            return new_clusters

        def calc_group_delay() -> np.ndarray:
            """
            set
            M
            """
            self.M = np.zeros((self.G.shape[1], self.D.shape[1]))

            # get clients in each group
            grouping = [ [] for i in range(self.G.shape[1])]
            for i, client in enumerate(self.G):
                for j, to_group in enumerate(client):
                    if self.G[i][j] == 1:
                        grouping[j].append(i)

            for i, group in enumerate(self.M):
                for j, to_server in enumerate(group):
                    # get group i delay to server j
                    max_delay = 0
                    for client in grouping[i]:
                        if self.D[client][j] > max_delay:
                            max_delay = self.D[client][j]
                    self.M[i][j] = max_delay

        # group clients by delay to servers
        groups: list[list[int]] = [ [] for i in range(self.D.shape[1]) ]
        # clients to their nearest servers
        server_indices = np.argmin(self.D, 1)
        for i, server in enumerate(server_indices):
            groups[server].append(i)

        group_num = 0
        clusters_list = []
        # cluster clients to the same server
        for group in groups:
            clusters = clustering(group)
            clusters = regroup(clusters, self.config.min_group_size)
            clusters_list.append(clusters)
            group_num += len(clusters)

        self.G = np.zeros((len(self.d), group_num), int)
        # G_size = np.zeros((group_num,))
        # M = -1 * np.ones((group_num, D.shape[1]))

        group_counter = 0
        for server, clusters in enumerate(clusters_list):
            for cluster in clusters:
                for client in cluster:
                    self.G[client][group_counter] = 1

                group_counter += 1
        
        calc_group_delay()

    def group_selection(self):
        """
        set
        A: g*s, group assignment matrix
        """
        def calc_loss_by_group()-> np.ndarray:
            group_loss = np.zeros((self.G.shape[1]))

            self.set_all_groups()

            for i, group in enumerate(self.all_groups):
                for client_index in group:
                    client = self.clients[client_index]
                    model = client.model
                    device = client.device
                    loss_fn = client.loss_fn
                    
                    model.to(device)
                    model.eval()

                    for (image, label) in client.trainloader:

                        y = model(image.to(device))
                        group_loss[i] += loss_fn(y, label.to(device)).item()

            group_loss /= self.all_groups_size

            return group_loss

        def calc_stds() -> np.ndarray:
            """
            return 
            std of number of all data in groups
            """

            stds = np.zeros((self.G.shape[1],))
            label_num = len(get_targets_set(self.d[0].dataset))
            label_list = np.zeros((self.G.shape[1], label_num))

            targets = get_targets(self.d[0].dataset)
            for i, client in enumerate(self.G):
                for j, to_group in enumerate(client):
                    if self.G[i][j] == 1:
                        for index in self.d[i].indices:
                            label_list[j][targets[index]] += 1

            for i, group in enumerate(label_list):
                total_num = np.sum(group)
                avg = total_num / len(group)
                for j, target in enumerate(group):
                    stds[i] += (label_list[i][j] - avg) ** 2 / len(group)

            return stds

        losses = calc_loss_by_group()
        stds = calc_stds()

        # rank all groups by quality
        Q = np.exp(1 + losses * 10) - np.log(stds + 1)
        seq = [ i for i in range(Q.shape[0])]
        Q_sorted = sorted(zip(Q, seq))
        # from greater to smaller
        Q_sorted.reverse()

        group_size_by_client = np.zeros((self.G.shape[1],))
        for i, client in enumerate(self.G):
            for j, to_group in enumerate(client):
                if self.G[i][j] == 1:
                    group_size_by_client[j] += 1

        B_temp = np.zeros((self.B.shape[0],))
        self.A = np.zeros(self.M.shape)
        group_assigned = np.zeros(self.A.shape[0])
        for (quality, seq) in Q_sorted:
            for j, to_server in enumerate(self.A[seq]):
                if B_temp[j] + group_size_by_client[seq] <= self.B[j] and self.M[seq][j] <= self.config.l and group_assigned[seq] == 0:
                    B_temp[j] += group_size_by_client[seq]
                    self.A[seq][j] = 1
                    group_assigned[seq] = 1
                else:
                    self.A[seq][j] = 0

        self.set_selected_groups()
        print("Number of data on selected clients: %d" % (sum(self.selected_groups_size),))
        print(np.exp(losses*10 + 1)[:10])
        print(np.log(stds + 1)[:10])

    def global_distribute(self) -> None:
        for client in self.clients:
            new_sd = deepcopy(self.model.state_dict())
            client.model.load_state_dict(new_sd)

    def global_train(self) -> float:
        """
        update self.model
        """
        def single_group_train(group: 'list[int]') -> float:
            """
            return loss
            """
            def group_train(group: 'list[int]') -> float:
                group_loss = 0
                # train all clients in this group
                for client_index in group:
                    client = self.clients[client_index]
                    group_loss += client.train()
                        
                return group_loss
                
            def group_aggregation(group: 'list[int]'):
                # get clients sizes
                C_size = []
                for index in group:
                    size = len(self.clients[index].trainset.indices)
                    C_size.append(size)
                data_num_sum = sum(C_size)

                # get state dicts
                state_dicts = []
                for index in group:
                    client = self.clients[index]
                    state_dicts.append(client.model.state_dict())

                # calculate average model
                state_dict_avg = deepcopy(state_dicts[0]) 
                for key in state_dict_avg.keys():
                    state_dict_avg[key] = 0 # state_dict_avg[key] * -1
                for key in state_dict_avg.keys():
                    for i in range(len(state_dicts)):
                        state_dict_avg[key] += state_dicts[i][key] * (C_size[i] / data_num_sum)
                
                # update all clients in this group
                selected_clients: list[Client] = [self.clients[i] for i in group]
                for i, client in enumerate(selected_clients):
                    new_sd = deepcopy(state_dict_avg)
                    client.model.load_state_dict(new_sd)

            
            for i in range(self.config.group_epoch_num):
                group_loss = group_train(group)
                group_aggregation(group)

            return group_loss

        # selected_groups, G_size = self.get_selected_groups()
        # print("Number of data on selected clients: %d" % (sum(G_size),))

        self.global_distribute()

        # compare_models(model, clients, 1)
        
        epoch_loss: float = 0
        for i, group in enumerate(self.selected_groups):
            epoch_loss += single_group_train(group)
        # average loss
        data_num = sum(self.selected_groups_size)
        epoch_loss /= data_num

        # compare_models(model, clients, 1)

        self.global_aggregation()

        # compare_models(model, clients, 1)

        return epoch_loss

    def global_aggregation(self) -> nn.Module:
        # selected_groups, groups_size = self.get_selected_groups()
        
        # G_size = get_groups_size(clients, G)
        data_num_sum = sum(self.selected_groups_size)

        # state dicts of the first client in each group
        state_dicts = []
        for i, group in enumerate(self.selected_groups):
            client = self.clients[group[0]]
            model = client.model.to(client.device)
            state_dicts.append(model.state_dict())

        # calculate average model
        state_dict_avg = deepcopy(state_dicts[0]) 
        for key in state_dict_avg.keys():
            state_dict_avg[key] = 0 # state_dict_avg[key] * -1

        for key in state_dict_avg.keys():
            for i in range(len(state_dicts)):
                state_dict_avg[key] += state_dicts[i][key] * (self.selected_groups_size[i] / data_num_sum)
            # state_dict_avg[key] = torch.div(state_dict_avg[key], len(state_dicts))
        
        self.model.load_state_dict(state_dict_avg)


    # def group2list(G: np.ndarray) -> 'list[list[int]]':
    #     groups: list[list[int]] = []
    #     G_T = G.transpose()
    #     for i in range(G.shape[1]):
    #         new_group = []
    #         for j, client in enumerate(G_T[i]):
    #             if client == 1:
    #                 new_group.append(j)

    #         groups.append(new_group)
        
    #     return groups



