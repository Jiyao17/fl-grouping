
from calendar import c
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset

import numpy as np


from enum import Enum
import copy
import random

from utils.data import TaskName, load_dataset, DatasetPartitioner, quick_draw
from utils.model import CIFARResNet, test_model

class Config:

    class SelectionMode(Enum):
        # prob based, smaller than random
        PROB_RCV = 1
        PROB_SRCV = 2
        PROB_ERCV = 3
        PROB_ESRCV = 4
        PROB_RCV_COST = 5
        PROB_WEIGHT = 9

        RANDOM = 10
        # ranking based, greater than random

    class GroupingMode(Enum):
        NONE = 1
        RANDOM = 2
        CV_GREEDY = 3
        NONIID = 4

    
    class TrainMethod(Enum):
        SGD = 1
        SCAFFOLD = 2
        FEDPROX = 3
        
    class AggregationOption(Enum):
        WEIGHTED_AVERAGE = 1
        UNBIASED = 2

    def __init__(self, task_name=TaskName.CIFAR,
        server_num=10, client_num=500, data_num_range=(10, 50), alpha=0.1,
        sampling_frac=0.2, budget=10**6,
        global_epoch_num=500, group_epoch_num=1, local_epoch_num=5,
        lr=0.1, lr_interval=100, local_batch_size=10,
        log_interval=5, 
        grouping_mode=GroupingMode.CV_GREEDY, max_group_cv=1.0, min_group_size=10,
        # partition_mode=PartitionMode.IID,
        selection_mode=SelectionMode.RANDOM, aggregation_option=7,
        device="cuda", 
        train_method = TrainMethod.SGD,
        result_dir="./exp_data/",
        test_mark: str="",
        comment="",
        data_path="../data/", 
        ) -> None:

        # federated learning system settings
        self.task_name = task_name
        self.server_num = server_num
        self.client_num = client_num
        self.data_num_range = data_num_range
        self.alpha = alpha
        self.sampling_frac = sampling_frac
        self.data_path = data_path
        self.budget = budget
        self.global_epoch_num = global_epoch_num
        self.group_epoch_num = group_epoch_num
        self.local_epoch_num = local_epoch_num
        self.lr_interval = lr_interval
        self.lr = lr
        self.batch_size = local_batch_size
        self.device = device
        # SCAFFOLD or not
        self.train_method = train_method

        self.selection_mode = selection_mode
        self.aggregation_option: Config.AggregationOption = aggregation_option
        # self.partition_mode = partition_mode
        self.grouping_mode = grouping_mode
        self.max_group_cv = max_group_cv
        self.min_group_size = min_group_size
        # results
        self.log_interval = log_interval
        self.result_dir = result_dir
        self.test_mark = test_mark
        
        self.comment = comment
    
    def change_output_file(self, num: int):
        self.result_file_accu = self.result_file_accu[0:-1] + str(num)
        self.result_file_loss = self.result_file_loss[0:-1] + str(num)


class Client:
    @staticmethod
    def init_clients(subsets_indexes: 'list[Subset]', lr, local_epoch_num, train_method, device, batch_size, loss=nn.CrossEntropyLoss()) \
        -> 'tuple[nn.Module, list[torch.Tensor], list[Client]]':
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

        # global c in SCAFFOLD
        c: list[torch.Tensor] = [ param.clone().zero_().to(device) for param in model.parameters()]


        for i in range(client_num):
            new_model = CIFARResNet()
            # torch.nn.init.normal_(model)
            sd = copy.deepcopy(model.state_dict())
            new_model.load_state_dict(sd)

            clients.append(Client(new_model, subsets_indexes[i], lr, local_epoch_num, train_method, device, batch_size, loss))

        return model, c, clients

    def __init__(
        self, model: nn.Module, 
        trainset: Subset, 
        lr: float, 
        local_epoch_num=5,
        train_method = Config.TrainMethod.SGD,
        device: str="cpu", 
        batch_size: int=0, 
        loss_fn=nn.CrossEntropyLoss()
        ) -> None:

        self.model = model
        self.trainset = trainset
        # self.training_cost = self.calc_training_cost(len(self.trainset.indices))
        self.lr = lr
        self.local_epoch_num = local_epoch_num
        self.train_method  = train_method
        self.device = device
        self.batch_size = batch_size if batch_size != 0 else len(trainset.indices)
        self.loss_fn = loss_fn

        self.trainloader = DataLoader(self.trainset, self.batch_size, True) #, drop_last=True
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9, weight_decay=0.0001) #

        # training results
        self.train_loss = 0
        self.grad: list[torch.Tensor] = [ param.clone().to(self.device) for param in self.model.parameters()]
        self.temp_model_params: list[torch.Tensor] = [ param.clone().zero_().to(self.device) for param in self.model.parameters()]
        self.c_client: list[torch.Tensor] = [ param.clone().zero_().to(self.device) for param in self.model.parameters()]
        self.c_delta: list[torch.Tensor] = [ param.clone().zero_().to(self.device) for param in self.model.parameters()]
        # self.c_client = [param.zero_() for param in self.c_client]
        self.c_global: list[torch.Tensor] = [ param.clone().zero_().to(self.device) for param in self.model.parameters()]

    def train(self):

        self.model.to(self.device)
        self.model.train()


        for i, param in enumerate(self.model.parameters()):
            self.temp_model_params[i] = param.detach().data

        # for i, c_d in enumerate(self.c_client):
        #     self.c_global[i] = c_d.clone().detach().data

        # reset loss and average gradient
        self.train_loss: float = 0
        for tensor in self.grad:
            tensor.zero_()

        for i in range(self.local_epoch_num):
            for (image, label) in self.trainloader:
                y = self.model(image.to(self.device))
                loss = self.loss_fn(y, label.to(self.device))

                if self.train_method == Config.TrainMethod.FEDPROX:
                    for w_t, w in zip(self.temp_model_params, self.model.parameters()):
                        loss += 1 / 2. * torch.pow(torch.norm(w.data - w_t.data), 2)

                self.train_loss += loss.item()

                self.optimizer.zero_grad()
                loss.backward()

                if self.train_method == Config.TrainMethod.SCAFFOLD:
                    for param, c_i, c_g in zip(self.model.parameters(), self.c_client, self.c_global):
                        param.grad += -c_i + c_g
                # get gradient 
                # for i, param in enumerate(self.model.parameters()):
                #     self.grad[i] += param.grad.detach().data

                self.optimizer.step()

        # for c_temp, param_c, param_g, c_i, c_g in zip(self.c_temp, self.model.parameters(), self.temp_model_params, self.c_client, self.c_global):

        #     c_temp = c_i - c_g + 1/(self.lr * self.local_epoch_num) * (param_g - param_c)

        if self.train_method == Config.TrainMethod.SCAFFOLD:
            for i, param_c in enumerate(self.model.parameters()):
                self.c_delta[i] = -1 * self.c_global[i] + (1/(self.lr * self.local_epoch_num)) * (self.temp_model_params[i] - param_c)

            for i, c_d in enumerate(self.c_delta):
                self.c_client[i] = self.c_client[i] - c_d
        # for i, param in enumerate(self.model.parameters()):
        #     self.grad[i] /= self.local_epoch_num * len(self.trainset.indices)
        # self.train_loss /= self.local_epoch_num * len(self.trainset.indices)
        return self.train_loss, self.grad

    def set_lr(self, new_lr):
        self.lr = new_lr
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9, weight_decay=0.0001) #

    # def calc_training_cost(self, dataset_len: int) -> float:
    #     training_cost = 0.00495469 * dataset_len + 0.01023199
    #     return training_cost

class GFL:
    def set_seed(self, seed=None):
        if seed is None:
            random.seed()
        else:
            random.seed(seed)

        seed = random.randint(1, 10000)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False
    
    def __init__(self, config: Config, data_seed=0) -> None:
        
        self.config = copy.deepcopy(config)

        self.faccu = open(self.config.result_dir + "accu" + self.config.test_mark, "a")
        self.floss = open(self.config.result_dir + "loss" + self.config.test_mark, "a")
        self.fcost = open(self.config.result_dir + "cost" + self.config.test_mark, "a")
        self.faccu.write("\nconfig:" + str(vars(self.config)) + "\n")
        self.floss.write("\nconfig:" + str(vars(self.config)) + "\n")
        self.fcost.write("\nconfig:" + str(vars(self.config)) + "\n")
        self.faccu.flush()
        self.floss.flush()
        self.fcost.flush()

        self.set_seed(data_seed)
        self.trainset, self.testset = load_dataset(self.config.task_name)

        partitioner = DatasetPartitioner(self.trainset, self.config.client_num, self.config.data_num_range, self.config.alpha)
        self.label_type_num = partitioner.label_type_num
        self.distributions = partitioner.get_distributions()
        self.subsets_indices = partitioner.get_subsets()
        # partitioner.draw(20,"./pic/dubug.png")
        # global model, global drift in SCAFFOLD, clients
        self.model, self.c_global, self.clients = Client.init_clients(self.subsets_indices, self.config.lr, self.config.local_epoch_num, self.config.train_method, self.config.device, self.config.batch_size)
        self.c_global : list[torch.Tensor] = [ param.clone().detach().zero_().to(self.config.device) for param in self.model.parameters()]

        self.clients_data_nums = np.sum(self.distributions, axis=1)
        self.clients_weights = self.clients_data_nums / np.sum(self.distributions)
        
        self.set_seed()

        # assign clients to servers
        self.servers_clients: 'list[list[int]]' = []
        indices = list(range(self.config.client_num))
        client_per_server = self.config.client_num // self.config.server_num
        for i in range(self.config.server_num):
            self.servers_clients.append(indices[i*client_per_server:(i+1)*client_per_server])
        # only modified in group() once
        self.groups: 'list[list[int]]'= []
        self.groups_data_nums: 'list[int]'= []
        self.groups_data_nums_arr: np.ndarray = None
        self.groups_weights: 'list[int]'= []
        self.groups_weights_arr: np.ndarray = None
        self.groups_cvs: 'list[int]'= []
        self.groups_cvs_arr: np.ndarray = None
        # weights for weighted random sampling without replacement
        self.probs_arr: np.ndarray = None
        # frequencys that group g is sampled, used to estaimate real probs
        # self.freqs_arr: np.ndarray = None

        self.groups_costs_arr: np.ndarray = None
        # assign groups to servers
        self.servers_groups: 'list[list[int]]'= [[] for _ in range(self.config.server_num)]
        #
        self.sampling_num = 0
        # may change over iterations
        self.selected_groups: np.ndarray = None


        self.group()
        pic_filename = self.config.result_dir + "group_distribution_" + self.config.test_mark + ".png"
        self.inspect_group_distribution(self.groups, len(self.groups)//2, pic_filename)

    def __calc_group_cv(self, subset_indices: 'list[int]') -> float:
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

    def inspect_group_distribution(self, groups: 'list[list[int]]', num: int=None, filename:str="./pic/group_distribution.png") -> None:
        """
        return 
        distribution: np.ndarray, (label_type_num,)
        """
        groups_distrs: np.ndarray = np.zeros((len(groups), self.distributions.shape[1]))
        for i, group in enumerate(groups):
            for j, client_index in enumerate(group):
                groups_distrs[i] += self.distributions[client_index]

        if num is None:
            num = len(self.distributions)

        DatasetPartitioner.plot_distribution(groups_distrs, num, filename)

    def group(self):
        """
        form groups
        record their sigmas
        """
        def __CV_greedy_grouping(server_clients_arg: 'list[int]', server_num, max_cv: float, min_group_size: int):
            """
            stop whichever sigma or group size is reached
            server_clients: list[int]
            return
            groups of this server
            """
            group_num_start = len(self.groups)
            server_clients = copy.deepcopy(server_clients_arg)
            # form groups for each server
            while len(server_clients) > 0:
                # try to form a new group

                # find a random client as the first one in the group
                cur_min_cv = self.__calc_group_cv([server_clients[0]])
                new_group: 'list[int]' = [server_clients[0]]
                # greedy for the first client
                # for client in server_clients:
                #     cv = self.__calc_group_cv([client])
                #     if cv < cur_min_cv:
                #         cur_min_cv = cv
                #         new_group = [client]
                server_clients.remove(new_group[0])

                # try to add more clients to the group
                while len(server_clients) > 0 and not (cur_min_cv < max_cv and len(new_group) >= min_group_size):
                    new_client = -1
                    # find the greedily best one
                    for client in server_clients:
                        new_group.append(client)
                        new_cv = self.__calc_group_cv(new_group)
                        if new_cv < cur_min_cv:
                            cur_min_cv = new_cv
                            new_client = client
                            
                        new_group.pop()

                    # no suitable client found, stop adding clients to the current group
                    if new_client == -1:
                        break
                    # found a suitable client, remove it from the pool, add it to the group
                    else:
                        server_clients.remove(new_client)
                        new_group.append(new_client)

                # not enough clients to form a group
                if len(new_group) < min_group_size:
                    continue
                else:
                    self.groups.append(new_group)
                    self.groups_data_nums.append(np.sum(self.distributions[new_group]))
                    self.groups_cvs.append(cur_min_cv)
                    self.servers_groups[server_num].append(group_num_start)
                    group_num_start += 1

        def __random_grouping(server_clients_arg: 'list[int]', server_no, min_group_size: int):
            """
            stop when group size is reached
            server_clients: list[int]
            return
            groups of this server
            """
            group_num_start = len(self.groups)
            server_clients = copy.deepcopy(server_clients_arg)
            # form groups for each server
            while len(server_clients) > 0:
                # try to form a new group

                new_group: 'list[int]' = []
                # try to add more clients to the group
                while len(server_clients) > 0 and len(new_group) < min_group_size:
                    new_client = np.random.choice(server_clients,size=1)[0]
                    # find the greedily best one
                    new_group.append(new_client)
                    server_clients.remove(new_client)

                # not enough clients to form a group
                if len(new_group) < min_group_size:
                    continue
                else:
                    self.groups.append(new_group)
                    self.groups_data_nums.append(np.sum(self.distributions[new_group]))
                    group_cv = self.__calc_group_cv(new_group)
                    self.groups_cvs.append(group_cv)
                    self.servers_groups[server_no].append(group_num_start)
                    group_num_start += 1
        
        def __calc_group_cost(groups: 'list[list[int]]') -> np.ndarray:
            # cost for one group train: local_cost * local epoch num + group aggregate cost * 1
            costs = np.zeros((len(groups),))
            
            for i, group in enumerate(groups):
                group_cost = 0
                
                group_size = len(group)
                for client_index in group:
                    # data from Raspberry PI 4
                    # secagg coefficiences: [ 0.01629675 -0.02373668  0.55442565]
                    # distance coefficiences: [ 0.00548707,  0.0038231,  -0.06900253]
                    # double param size (SCAFFOLD): [0.01879308 0.18775216 0.19883809]
                    # fedprox coefs: [0.06719291 0.14201339]

                    # secagg coeffs
                    group_coefs = [ 0.01629675, -0.02373668,  0.55442565]
                    # regular train coeffs
                    train_coefs = [ 0.07093414, -0.00559966]

                    if self.config.train_method == Config.TrainMethod.SCAFFOLD:
                        group_coefs = [0.01879308, 0.18775216, 0.19883809]
                    if self.config.train_method == Config.TrainMethod.FEDPROX:
                        train_coefs = [0.06719291, 0.14201339]

                    training_cost = (train_coefs[0] * self.clients_data_nums[client_index] + train_coefs[1]) * self.config.local_epoch_num
                    group_overhead = (group_coefs[0] * (group_size*group_size) + group_coefs[1] * group_size + group_coefs[2])

                    client_cost = (training_cost + group_overhead) * self.config.group_epoch_num
                    group_cost += client_cost
                
                costs[i] = group_cost
            
            return costs

        self.groups = []
        self.groups_cvs = []

        for i, server_clients in enumerate(self.servers_clients):
            if self.config.grouping_mode == Config.GroupingMode.CV_GREEDY:
                __CV_greedy_grouping(server_clients, i, self.config.max_group_cv, self.config.min_group_size)
            elif self.config.grouping_mode == Config.GroupingMode.RANDOM:
                __random_grouping(server_clients, i, self.config.min_group_size)
            elif self.config.grouping_mode == Config.GroupingMode.NONE:
                self.groups = [ [i] for i in range(len(self.clients))]
                self.groups_data_nums = copy.deepcopy(self.clients_data_nums)
                self.groups_cvs = [ self.__calc_group_cv(group) for group in self.groups]
                self.servers_groups = copy.deepcopy(self.servers_clients)

        self.groups_data_nums_arr = np.array(self.groups_data_nums)
        self.groups_weights_arr = self.groups_data_nums / np.sum(self.groups_data_nums)
        self.groups_cvs_arr = np.array(self.groups_cvs)
        self.groups_costs_arr = __calc_group_cost(self.groups)

        avg_group_size = np.mean(self.groups_data_nums_arr)
        sampling_data_num = self.config.sampling_frac * np.sum(self.clients_data_nums)
        self.sampling_num = int(sampling_data_num / avg_group_size)


    def global_distribute(self, selected_groups: 'list[int]'):
        """
        distribute model to selected groups
        """
        # distribute
        for group in selected_groups:
            for client in self.groups[group]:
                new_sd = copy.deepcopy(self.model.state_dict())
                self.clients[client].model.load_state_dict(new_sd)
                self.clients[client].c_global = [ c_g.clone().detach().data for c_g in self.c_global]

    def __calc_probs(self) -> np.ndarray:
        """
        for probability based selection
        calculate the probability of each group
        """
        probs: np.ndarray = None
        if self.config.selection_mode == Config.SelectionMode.RANDOM:
            probs = np.full((len(self.groups), ), 1.0/len(self.groups), dtype=np.float)
        elif self.config.selection_mode == Config.SelectionMode.PROB_RCV:
            probs = 1.0 / self.groups_cvs_arr
            # np.multiply(probs, self.groups_data_nums_arr, out=probs)
            sum_rcv = np.sum(probs)
            probs = probs / sum_rcv
        elif self.config.selection_mode == Config.SelectionMode.PROB_SRCV:
            probs = np.square(1.0 / self.groups_cvs_arr)
            sum_rcv = np.sum(probs)
            probs = probs / sum_rcv
        elif self.config.selection_mode == Config.SelectionMode.PROB_ERCV:
            probs = np.exp(1.0 / self.groups_cvs_arr)
            sum_rcv = np.sum(probs)
            probs = probs / sum_rcv
        elif self.config.selection_mode == Config.SelectionMode.PROB_ESRCV:
            probs = np.exp(np.square(1.0 / self.groups_cvs_arr))
            sum_rcv = np.sum(probs)
            probs = probs / sum_rcv
        elif self.config.selection_mode == Config.SelectionMode.PROB_RCV_COST:
            probs = 1.0 / np.multiply(self.groups_cvs_arr, self.groups_costs_arr)
            # np.multiply(probs, self.groups_data_nums_arr, out=probs)
            sum_rcv = np.sum(probs)
            probs = probs / sum_rcv
        elif self.config.selection_mode == Config.SelectionMode.PROB_WEIGHT:
            sum_sizes = np.sum(self.groups_data_nums_arr)
            probs = self.groups_data_nums_arr / sum_sizes


        self.probs_arr = probs
        return probs.copy()

    def sample(self) -> np.ndarray:
        """
        set self.selected_groups
        select groups for the current iteration
        list of group numbers (in self.groups)
        """
        self.selected_groups: np.ndarray = None

        if self.config.selection_mode.value <= Config.SelectionMode.RANDOM.value:
            probs = self.__calc_probs()
            # _pc = probs.copy()
            # np.multiply(probs, self.groups_data_nums_arr, out=_pc) # weighted average data number of each group
            # weighted_avg = np.sum(_pc)
            indices = range(len(self.groups))
            # sampling_num = int((sum(self.clients_data_nums) * self.config.sampling_frac) / weighted_avg)
            self.selected_groups = np.random.choice(indices, self.sampling_num, p=probs, replace=False)
        
        else:
            pass

        # add more group if not enough

        return self.selected_groups

    def group_train(self, group: int):
        """
        return loss
        """
        def group_train_all_devices(group_index: int) -> float:
            group_loss = 0
            # train all clients in this group
            for client_index in self.groups[group_index]:
                client = self.clients[client_index]
                group_loss += client.train()[0]
                    
            return group_loss
            
        def group_aggregate_distribute(group_index: int):
            # get clients sizes
            # C_size = []
            # for index in self.groups[group]:
            #     size = len(self.clients[index].trainset.indices)
            #     C_size.append(size)
            # data_num_sum = sum(C_size)

            # init state dict
            client0 = self.clients[self.groups[group_index][0]]
            state_dict_avg = copy.deepcopy(client0.model.state_dict()) 
            c_group = [ c_g.clone().detach().data for c_g in client0.c_global]
            for key in state_dict_avg.keys():
                state_dict_avg[key] = 0.0

            # get average state dict and delta drift
            for client_index in self.groups[group_index]:
                state_dict = self.clients[client_index].model.state_dict()
                weight = self.clients_data_nums[client_index] / self.groups_data_nums[group_index]
                # weight = 1.0 / len(self.groups[group_index])
                for key in state_dict_avg.keys():
                    state_dict_avg[key] += state_dict[key] * weight

                if self.config.train_method == Config.TrainMethod.SCAFFOLD:
                    # delta drift in SCAFFOLD
                    for i, c_d in enumerate(c_group):
                        c_group[i] += self.clients[client_index].c_delta[i] * weight

            # update all clients in this group
            for client_index in self.groups[group_index]:
                new_sd = copy.deepcopy(state_dict_avg)
                self.clients[client_index].model.load_state_dict(new_sd)

            if self.config.train_method == Config.TrainMethod.SCAFFOLD:
                for client_index in self.groups[group_index]:
                    for i, c_d in enumerate(c_group):
                        self.clients[client_index].c_global[i] = c_group[i].clone().detach().data


        for i in range(self.config.group_epoch_num):
            group_train_all_devices(group)
            group_aggregate_distribute(group)

    def global_train(self, selected_groups: 'list[int]'):
        for i, group in enumerate(selected_groups):
            self.group_train(group)

    def global_aggregate(self, selected_groups: 'list[int]'):
        """
        under development
        aggregate model from selected groups
        """
        selected_groups_sizes_by_data = self.groups_data_nums_arr[selected_groups]
        selected_groups_data_sum = np.sum(selected_groups_sizes_by_data)
        total_data_sum = np.sum(self.groups_data_nums_arr)

        # init state dict
        client0 = self.clients[self.groups[selected_groups[0]][0]]
        state_dict_avg = copy.deepcopy(client0.model.state_dict()) 
        self.c_global = [ c_d.clone().zero_().detach().data for c_d in client0.c_delta]
        for key in state_dict_avg.keys():
            state_dict_avg[key] = 0.0


        weights = selected_groups_sizes_by_data / selected_groups_data_sum

        if self.config.aggregation_option == Config.AggregationOption.UNBIASED:
            weights = selected_groups_sizes_by_data / total_data_sum

            unbiased_factor = (self.probs_arr[selected_groups] * len(self.selected_groups))
        # factor_scale = 10.0
        # unbiased_factor[ unbiased_factor < 1/factor_scale ] = 1/factor_scale
        # unbiased_factor[ unbiased_factor > factor_scale ] = factor_scale
            weights = np.divide(weights, unbiased_factor)

            # if self.config.aggregation_option.value >= Config.AggregationOption.NUMERICAL_REGULARIZATION.value:
                # numerical adjustment, make training stable
            weights = weights / np.sum(weights)
        
        print(f"weights: {weights}")
                # print(f"unbiased weights: {weights[:10]}")

        # get average state dict
        for weight, group_index in zip(weights, selected_groups):
            repr_client = self.clients[self.groups[group_index][0]]
            state_dict = repr_client.model.state_dict()

            for key in state_dict_avg.keys():
                state_dict_avg[key] += state_dict[key] * weight

            # average delta drift in SCAFFOLD
            if self.config.train_method == Config.TrainMethod.SCAFFOLD:
                # delta drift in SCAFFOLD
                for i, c_d in enumerate(self.c_global):
                    self.c_global[i] += repr_client.c_client[i] * weight

        
        self.model.load_state_dict(state_dict_avg)


    def calc_selected_groups_cost(self, selected_groups: 'list[int]'):
        """
        return cost
        """
        cost = 0
        for group_index in selected_groups:
            cost += self.groups_costs_arr[group_index]
        return cost

    def run(self):
        # indices = np.random.choice(range(len(self.testset)), 1000, replace=False)
        # subtestset = Subset(self.testset, indices=indices)
        # # print(indices)
        self.testloader = DataLoader(self.testset, 1000, shuffle=True)

        # print(self.groups)
        accus = []
        losses = []
        costs = []
        cur_cost = 0
        for i in range(self.config.global_epoch_num):
            # lr decay
            if i % self.config.lr_interval == self.config.lr_interval - 1:
                for client in self.clients:
                    client.set_lr(client.lr / 10)
                print('lr decay to {}'.format(self.clients[0].lr))

            selected_groups = self.sample()
            selected_cost = self.calc_selected_groups_cost(selected_groups)

            # show_data_len = 10
            # if len(self.groups) < show_data_len:
            #     show_data_len = len(self.groups)
            print("costs", self.groups_costs_arr)
            print("probs", self.probs_arr)
            print("selected groups", self.selected_groups)
            group_sizes = [ len(group) for group in self.groups]
            print("[min, max] gs: ", np.min(group_sizes), np.max(group_sizes))
            print("mean gs, cv: ", np.mean(group_sizes), np.mean(self.groups_cvs_arr))
            data_selected = np.sum(self.groups_data_nums_arr[self.selected_groups])
            print('selected data num, cost:', data_selected, selected_cost)


            self.global_distribute(selected_groups)
            self.global_train(selected_groups)
            self.global_aggregate(selected_groups)

            # print("costs", self.groups_costs_arr)
            # print("probs", self.probs_arr)
            # print("selected", self.selected_groups)
            # group_sizes = [ len(group) for group in self.groups]
            # print("[min, max] gs: ", np.min(group_sizes), np.max(group_sizes))
            # print("mean gs, cv: ", np.mean(group_sizes), np.mean(self.groups_cvs_arr))
            # data_selected = np.sum(self.groups_data_nums_arr[self.selected_groups])
            cur_cost += selected_cost


            # test and record
            if i % self.config.log_interval == self.config.log_interval - 1:
                accu, loss = test_model(self.model, self.testloader)

                self.faccu.write(f'{accu} ')
                self.floss.write(f'{loss} ')
                self.fcost.write(f'{cur_cost} ')
                self.faccu.flush()
                self.floss.flush()
                self.fcost.flush()

                accus.append(accu)
                losses.append(loss)
                costs.append(cur_cost)

                quick_draw(accus, self.config.result_dir + 'accu' + str(self.config.test_mark) + '.png')
                print(f'epoch {i} accu: {accu} loss: {loss} cost: {cur_cost}')
                
            if cur_cost > self.config.budget:
                break
        self.faccu.close()
        self.floss.close()
        self.fcost.close()
    
