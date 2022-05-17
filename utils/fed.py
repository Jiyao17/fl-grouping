
from calendar import c
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset

import numpy as np


from enum import Enum
import copy

from utils.data import load_dataset, DatasetPartitioner
from utils.model import CIFARResNet

class Config:
    class Task(Enum):
        CIFAR = 1

    class SelectionMode(Enum):
        GRADIENT_RANKING = 1
        GRADIENT_PROB = 2
        RANDOM = 3
        GRADIENT_RANKING_RT = 4
        STD = 5
        LOW_STD_GRADIENT_RANKING_RT = 6
        LOW_STD_GRADIENT_RANKING = 7
        LOW_STD_RANDOM = 8

    class GroupingMode(Enum):
        CV_GREEDY = 1
        NONIID = 2
        RANDOM = 3
    
    class PartitionMode(Enum):
        IID = 1
        DIRICHLET = 2
        RANDOM = 3
        # IID_AND_NON_IID = 4

    def __init__(self, task_name="CIFAR",
        server_num=10, client_num=500, data_num_range=(10, 50), alpha=0.1,
        sampling_num=100,
        global_epoch_num=500, group_epoch_num=1, local_epoch_num=5,
        lr=0.1, lr_interval=100, local_batch_size=10,
        log_interval=5, 
        grouping_mode=GroupingMode.CV_GREEDY, cv=1, min_group_size=10,
        partition_mode=PartitionMode.IID,
        selection_mode=SelectionMode.GRADIENT_RANKING,
        device="cuda", 
        result_file_accu="./cifar/grouping/accu", 
        result_file_loss="./cifar/grouping/loss",
        comment="",
        data_path="../data/", 
        ) -> None:

        # federated learning system settings
        self.task_name = task_name
        self.server_num = server_num
        self.client_num = client_num
        self.data_num_range = data_num_range
        self.alpha = alpha
        self.sampling_num = sampling_num
        self.data_path = data_path
        self.global_epoch_num = global_epoch_num
        self.group_epoch_num = group_epoch_num
        self.local_epoch_num = local_epoch_num
        self.lr_interval = lr_interval
        self.lr = lr
        self.batch_size = local_batch_size
        self.device = device

        self.selection_mode = selection_mode
        self.partition_mode = partition_mode
        self.grouping_mode = grouping_mode
        self.max_cv = cv
        self.min_group_size = min_group_size
        # results
        self.log_interval = log_interval
        self.result_file_accu = result_file_accu
        self.result_file_loss = result_file_loss
        
        self.comment = comment
    
    def use_file(self, num: int):
        self.result_file_accu = self.result_file_accu[0:-1] + str(num)
        self.result_file_loss = self.result_file_loss[0:-1] + str(num)


class Client:
    @staticmethod
    def init_clients(subsets_indexes: 'list[Subset]', lr, local_epoch_num, device, batch_size, loss=nn.CrossEntropyLoss()) \
        -> 'tuple[nn.Module, list[Client]]':
        """
        return
        model: global model
        models: models on clients
        """
        clients: 'list[Client]' = []
        client_num = len(subsets_indexes)
        model: nn.Module = CIFARResNet()
        model.to(device)
        sd = model.state_dict()
        for key in sd.keys():
            if key.endswith('batches_tracked') is False:
                sd[key] = nn.init.normal_(sd[key], 0.0, 1.0)
        model.load_state_dict(sd)

        for i in range(client_num):
            new_model = CIFARResNet()
            # torch.nn.init.normal_(model)
            sd = copy.deepcopy(model.state_dict())
            new_model.load_state_dict(sd)

            clients.append(Client(new_model, subsets_indexes[i], lr, local_epoch_num, device, batch_size, loss))

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

        self.trainloader = DataLoader(self.trainset, self.batch_size, True) #, drop_last=True
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9, weight_decay=0.0001) #

        # training results
        self.train_loss = 0
        self.grad: list[torch.Tensor] = [ param.clone().to(self.device) for param in self.model.parameters()]

    def train(self):
        self.model.to(self.device)
        self.model.train()

        # reset loss and average gradient
        self.train_loss = 0
        for tensor in self.grad:
            tensor.zero_()

        for i in range(self.local_epoch_num):
            for (image, label) in self.trainloader:
                y = self.model(image.to(self.device))
                loss = self.loss_fn(y, label.to(self.device))
                self.train_loss += loss.item()

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                # get gradient 
                for i, param in enumerate(self.model.parameters()):
                    self.grad[i] += param.grad.detach().data
        
        for i, param in enumerate(self.model.parameters()):
            self.grad[i] /= self.local_epoch_num * self.batch_size
        self.train_loss /= self.local_epoch_num * self.batch_size
        return self.train_loss, self.grad

    def set_lr(self, new_lr):
        self.lr = new_lr
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9, weight_decay=0.0001) #


class GFL:
    
    def __init__(self, config: Config) -> None:
        self.config = copy.deepcopy(config)

        # self.faccu = open(self.config.result_file_accu, "a")
        # self.floss = open(self.config.result_file_loss, "a")
        # self.faccu.write("\nconfig:" + str(vars(self.config)) + "\n")
        # self.floss.write("\nconfig:" + str(vars(self.config)) + "\n")
        # self.faccu.flush()
        # self.floss.flush()

        self.trainset, self.testset = load_dataset(self.config.task_name)
        partitioner = DatasetPartitioner(self.trainset, self.config.client_num, self.config.data_num_range, self.config.alpha)
        self.label_type_num = partitioner.label_type_num
        self.distributions = partitioner.get_distributions()
        self.subsets_indices = partitioner.get_subsets()
        # partitioner.draw(20,"./pic/dubug.png")
        self.testloader = DataLoader(self.testset, 1000, drop_last=True)
        self.model, self.clients = Client.init_clients(self.subsets_indices, self.config.lr, self.config.local_epoch_num, self.config.device, self.config.batch_size)

        # assign clients to servers
        self.servers_clients: 'list[list[int]]' = []
        indices = list(range(self.config.client_num))
        client_per_server = self.config.client_num // self.config.server_num
        for i in range(self.config.server_num):
            self.servers_clients.append(indices[i*client_per_server:(i+1)*client_per_server])
        # only modified in group() once
        self.groups: 'list[int]'= []
        # assign groups to servers
        self.servers_groups: 'list[list[int]]'= []
        # may change over iterations
        self.selected_groups = []

        self.group()

    def calc_cv(self, subset_indices: 'list[int]') -> float:
        """
        optimizable
        return 
        cv: float, coefficient of variation
        """
        distribution = np.zeros((self.label_type_num,))
        for index in subset_indices:
            distribution += self.distributions[index]
        cv = np.std(distribution) / np.mean(distribution)
        return cv

    def group(self):
        """
        form groups
        record their sigmas
        """
        def CV_greedy(server_clients_arg: 'list[int]', server_no, max_cv: float, min_group_size: int) -> 'list[list[int]]':
            """
            stop whichever sigma or group size is reached
            server_clients: list[int]
            return
            groups of this server
            """
            group_no_start = len(self.groups)
            server_clients = copy.deepcopy(server_clients_arg)
            while len(server_clients) > 0:
                # find the client with min cv
                cur_min_cv = self.calc_cv([server_clients[0]])
                new_group: 'list[int]' = [server_clients[0]]
                for client in server_clients:
                    cv = self.calc_cv([client])
                    if cv < cur_min_cv:
                        cur_min_cv = cv
                        new_group = [client]

                while cur_min_cv > max_cv and len(new_group) < min_group_size and len(server_clients) > 0:
                    new_client = -1
                    for client in server_clients:
                        new_group = new_group.append(client)
                        new_cv = self.calc_cv(new_group)
                        if new_cv < cur_min_cv:
                            cur_min_cv = new_cv
                            new_client = client
                            
                        new_group.pop()

                    server_clients.remove(new_client)
                    new_group.append(new_client)

                if len(new_group) < min_group_size:
                    continue
                else:
                    self.groups.append(new_group)
                    self.groups_cvs.append(cur_min_cv)
                    self.servers_groups[server_no].append(group_no_start)
                    group_no_start += 1

        self.groups = []
        self.groups_cvs = []

        for i, server_clients in enumerate(self.servers_clients):
            if self.config.grouping_mode == Config.GroupingMode.CV_GREEDY:
                CV_greedy(server_clients, i, self.config.max_cv, self.config.min_group_size)


    def distribute(self, selected_groups: 'list[int]'):
        """
        distribute model to selected groups
        """
        pass

    def calc_probs(self) -> np.ndarray:
        """
        calculate the probability of each group
        """
        probs = np.full((self.config.client_num, ), 1.0/len(self.groups), dtype=np.float)

        return probs

    def sample(self) -> 'list[int]':
        """
        select groups for the current iteration
        list of group numbers (in self.groups)
        """
        probs = np.zeros((len(self.groups),), dtype=np.float)
        for i, group in enumerate(self.groups):
            probs[i] = 1.0 / self.calc_cv(group)
        sum_rcv = np.sum(probs)
        probs = probs / sum_rcv

        indices = range(len(self.groups))
        self.selected_groups= np.random.choice(indices, self.config.sampling_num, p=probs)
        return self.selected_groups

    def train(self, selected_groups: 'list[int]'):
        pass

    def aggregate(self, selected_groups: 'list[int]'):
        """
        aggregate model from selected groups
        """
        pass

    def run(self):
        for i in range(self.config.global_epoch_num):
            selected_groups = self.sample()
            self.distribute(selected_groups)
            self.train(selected_groups)
            self.aggregate(selected_groups)



