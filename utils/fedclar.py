
from asyncio import Task
from matplotlib import test
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset

import numpy as np
import os

from enum import Enum
import copy
import random

from utils.data import TaskName, load_dataset, DatasetPartitioner, quick_draw
from utils.model import CIFARResNet, test_model, SpeechCommand

from torch.utils.data.dataset import Dataset, Subset

from torchaudio.datasets import SPEECHCOMMANDS
from torchaudio.transforms import Resample
import torch.nn.functional as F
from torch import nn, optim, Tensor


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
        FEDCLAR = 5

    
    class TrainMethod(Enum):
        SGD = 1
        SCAFFOLD = 2
        FEDPROX = 3
        FEDCLAR = 4
        
    class AggregationOption(Enum):
        WEIGHTED_AVERAGE = 1
        UNBIASED = 2

    def __init__(self, task_name=TaskName.CIFAR,
        server_num=10, client_num=500, data_num_range=(10, 50), alpha=0.1,
        sampling_frac=0.2, budget=10**6, FedCLAR_cluster_epoch=30, FedCLAR_tl_epoch=50, FedCLAR_th=0.1,
        global_epoch_num=500, group_epoch_num=1, local_epoch_num=5,
        lr=0.1, lr_interval=100, local_batch_size=10,
        log_interval=5, 
        grouping_mode=GroupingMode.CV_GREEDY, max_group_cv=1.0, min_group_size=10,
        # partition_mode=PartitionMode.IID,
        selection_mode=SelectionMode.RANDOM, aggregation_option=AggregationOption.WEIGHTED_AVERAGE,
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
        self.FedCLAR_cluster_epoch= FedCLAR_cluster_epoch
        self.FedCLAR_tl_epoch = FedCLAR_tl_epoch
        self.FedCLAR_th = FedCLAR_th
        self.group_epoch_num = group_epoch_num
        self.local_epoch_num = local_epoch_num
        self.lr_interval = lr_interval
        self.lr = lr
        self.batch_size = local_batch_size
        self.device = device
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


class TaskSpeechCommand():

    labels: list = ['backward', 'bed', 'bird', 'cat', 'dog', 'down', 'eight', 'five', 'follow',
        'forward', 'four', 'go', 'happy', 'house', 'learn', 'left', 'marvin', 'nine', 'no', 'off',
        'on', 'one', 'right', 'seven', 'sheila', 'six', 'stop', 'three', 'tree', 'two', 'up', 
        'visual', 'wow', 'yes', 'zero']

    class SubsetSC(SPEECHCOMMANDS):
        def __init__(self, subset, data_path):
            super().__init__(root=data_path, download=True)

            def load_list(filename):
                filepath = os.path.join(self._path, filename)
                with open(filepath) as fileobj:
                    return [os.path.join(self._path, line.strip()) for line in fileobj]

            if subset == "validation":
                self._walker = load_list("validation_list.txt")
            elif subset == "testing":
                self._walker = load_list("testing_list.txt")
            elif subset == "training":
                excludes = load_list("validation_list.txt") + load_list("testing_list.txt")
                excludes = set(excludes)
                self._walker = [w for w in self._walker if w not in excludes]

    def __init__(self, config: Config, trainset=None, testset=None):
        self.config = copy.deepcopy(config)
        self.trainset = trainset
        self.testset = testset

        self.scheduler = None
        self.loss_fn = F.nll_loss

        # print(len(self.trainset))
        if self.testset is not None:
            waveform, sample_rate, label, speaker_id, utterance_number = self.testset[0]
        else:
            waveform, sample_rate, label, speaker_id, utterance_number = self.trainset[0]
        new_sample_rate = 8000
        transform = Resample(orig_freq=sample_rate, new_freq=new_sample_rate)
        # transformed: Resample = transform(waveform)
        self.transform = transform.to(self.config.device)
        waveform = waveform.to(self.config.device)
        self.tranformed = self.transform(waveform).to(self.config.device)
        self.model = SpeechCommand(n_input=self.tranformed.shape[0], n_output=len(self.labels)).to(self.config.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.lr, weight_decay=0.0001)
        step_size = self.config.lr_interval * self.config.group_epoch_num * self.config.local_epoch_num
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=step_size, gamma=0.5)  # reduce the learning after 20 epochs by a factor of 10

        if trainset is not None:
            self.train_dataloader = TaskSpeechCommand.get_dataloader("train", self.config, trainset)
        if testset is not None:
            self.test_dataloader = TaskSpeechCommand.get_dataloader("test", self.config, None, testset)

    @staticmethod
    def get_datasets(config: Config) -> 'tuple[Dataset, Dataset]':
        testset = TaskSpeechCommand.SubsetSC("testing", config.data_path)
        trainset = TaskSpeechCommand.SubsetSC("training", config.data_path)

        removed_train = []
        removed_test = []
        for i in range(len(trainset)):
            waveform, sample_rate, label, speaker_id, utterance_number = trainset[i]
            if waveform.shape[-1] != 16000:
                removed_train.append(i)

        trainset = Subset(trainset, list(set(range(len(trainset))) - set(removed_train)))

        for i in range(len(testset)):
            waveform, sample_rate, label, speaker_id, utterance_number = testset[i]
            if waveform.shape[-1] != 16000:
                removed_test.append(i)
                # testset._walker.remove(testset._walker[i])
                # removed_test += 1
        
        testset = Subset(testset, list(set(range(len(testset))) - set(removed_test)))

        print("Data number removed from trainset: ", len(removed_train))
        print("Data number removed from testset: ", len(removed_test))
        return (trainset, testset)

    @staticmethod
    def get_dataloader(loader, config: Config, trainset=None, testset=None, ):

        if config.device == torch.device("cuda"):
            num_workers = 1
            pin_memory = True
        else:
            num_workers = 0
            pin_memory = False

        # test dataloader
        if loader != "train":
            test_dataloader = DataLoader(
                    testset,
                    batch_size=1000,
                    shuffle=False,
                    drop_last=True,
                    collate_fn=TaskSpeechCommand.collate_fn,
                    num_workers=num_workers,
                    pin_memory=pin_memory,
                    )
        if loader != "test":
        # train dataloader
        # if 0 <= self.config.reside and self.configs.reside <= self.configs.client_num:
        #     data_num = self.configs.l_data_num
        #     reside = self.configs.reside
        #     self.trainset = Subset(Task.trainset,
        #         Task.trainset_perm[data_num*reside: data_num*(reside+1)])
            # self.trainset = trainset
            train_dataloader = DataLoader(
            trainset,
            batch_size=config.batch_size,
            shuffle=True,
            collate_fn=TaskSpeechCommand.collate_fn,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False
            )
            
        if loader == "train":
            return train_dataloader
        elif loader == "test":
            return test_dataloader
        else:
            return train_dataloader, test_dataloader

        

    def train(self):
        self.model.to(self.config.device)
        self.model.train()
        self.transform = self.transform.to(self.config.device)
        for data, target in self.train_dataloader:
            data = data.to(self.config.device)
            target = target.to(self.config.device)
            # apply transform and model on whole batch directly on device
            data = self.transform(data)
            output = self.model(data)
            # negative log-likelihood for a tensor of size (batch x 1 x n_output)
            loss = self.loss_fn(output.squeeze(), target)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            # self.scheduler.step()

    def test(self, test_dataloader):
        self.model.to(self.config.device)
        self.model.eval()

        dataset_size = len(test_dataloader.dataset)
        correct, loss = 0, 0
        for data, target in test_dataloader:
            data = data.to(self.config.device)
            target = target.to(self.config.device)
            # apply transform and model on whole batch directly on device
            data = self.transform(data)
            output = self.model(data)

            pred = TaskSpeechCommand.get_likely_index(output)
            loss += self.loss_fn(output.squeeze(), target).item()

            # pred = output.argmax(dim=-1)
            correct += TaskSpeechCommand.number_of_correct(pred, target)

        correct /= 1.0*dataset_size
        loss /= 1.0*dataset_size

        return correct, loss

    @staticmethod
    def label_to_index(word):
        # Return the position of the word in labels
        return torch.tensor(TaskSpeechCommand.labels.index(word))

    @staticmethod
    def index_to_label(index):
        # Return the word corresponding to the index in labels
        # This is the inverse of label_to_index
        return TaskSpeechCommand.labels[index]

    @staticmethod    
    def pad_sequence(batch):
        # Make all tensor in a batch the same length by padding with zeros
        batch = [item.t() for item in batch]
        batch = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0.)
        return batch.permute(0, 2, 1)

    @staticmethod
    def collate_fn(batch):
        # A data tuple has the form:
        # waveform, sample_rate, label, speaker_id, utterance_number
        tensors, targets = [], []

        # Gather in lists, and encode labels as indices
        for waveform, _, label, *_ in batch:
            tensors += [waveform]
            targets += [TaskSpeechCommand.label_to_index(label)]

        # Group the list of tensors into a batched tensor
        tensors = TaskSpeechCommand.pad_sequence(tensors)
        targets = torch.stack(targets)

        return tensors, targets

    @staticmethod
    def number_of_correct(pred, target):
        # count number of correct predictions
        return pred.squeeze().eq(target).sum().item()

    @staticmethod
    def get_likely_index(tensor):
        # find most likely label index for each element in the batch
        return tensor.argmax(dim=-1)

    @staticmethod
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)



class Client:
    @staticmethod
    def init_clients(subsets_indexes: 'list[Subset]', config: Config) \
        -> 'tuple[nn.Module, list[torch.Tensor], list[Client]]':
        """
        return
        model: global model
        models: models on clients
        """
        
        clients: 'list[Client]' = []
        client_num = len(subsets_indexes)
        if config.task_name == TaskName.CIFAR:
            model: nn.Module = CIFARResNet()
        elif config.task_name == TaskName.SPEECHCOMMAND:
            model = SpeechCommand()

        model.to(config.device)
        sd = model.state_dict()
        for key in sd.keys():
            if key.endswith('batches_tracked') is False:
                sd[key] = nn.init.normal_(sd[key], 0.0, 1.0)
        model.load_state_dict(sd)

        # global c in SCAFFOLD
        c: list[torch.Tensor] = [ param.clone().zero_().to(config.device) for param in model.parameters()]


        for i in range(client_num):
            if config.task_name == TaskName.CIFAR:
                new_model = CIFARResNet()
            elif config.task_name == TaskName.SPEECHCOMMAND:
                new_model = SpeechCommand()
            # torch.nn.init.normal_(model)
            sd = copy.deepcopy(model.state_dict())
            new_model.load_state_dict(sd)
            # print(subsets_indexes)
            client = Client(new_model, subsets_indexes[i], None, config)
            clients.append(client)

        return model, c, clients

    def __init__(
        self, model: nn.Module, 
        trainset: Subset, 
        testset: Subset,
        config: Config = None
        ) -> None:
        self.config = copy.deepcopy(config)
        self.task_name = self.config.task_name
        self.trainset = trainset
        self.testset = testset
        if self.config.task_name == TaskName.SPEECHCOMMAND:
            self.task = TaskSpeechCommand(config, trainset)
            self.model = self.task.model
            self.trainloader = self.task.train_dataloader
            self.optimizer = self.task.optimizer
            self.scheduler = self.task.scheduler
            self.loss_fn = self.task.loss_fn
            self.transform = self.task.transform
            self.transform.to(self.config.device)

        elif self.config.task_name == TaskName.CIFAR:
            self.model = model
            # self.training_cost = self.calc_training_cost(len(self.trainset.indices))
            self.loss_fn = nn.CrossEntropyLoss()

            self.trainloader = DataLoader(self.trainset, self.config.batch_size, True) #, drop_last=True
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.config.lr, momentum=0.9, weight_decay=0.0001) #
        
        self.lr = self.config.lr
        self.local_epoch_num = self.config.local_epoch_num
        self.train_method  = self.config.train_method
        self.device = self.config.device
        self.batch_size = self.config.batch_size if self.config.batch_size != 0 else len(trainset.indices)

        # training results
        self.train_loss = 0
        self.grad: list[torch.Tensor] = [ param.clone().to(self.device) for param in self.model.parameters()]
        self.temp_model_params: list[torch.Tensor] = [ param.clone().zero_().to(self.device) for param in self.model.parameters()]
        self.c_client: list[torch.Tensor] = [ param.clone().zero_().to(self.device) for param in self.model.parameters()]
        self.c_delta: list[torch.Tensor] = [ param.clone().zero_().to(self.device) for param in self.model.parameters()]
        # self.c_client = [param.zero_() for param in self.c_client]
        self.c_global: list[torch.Tensor] = [ param.clone().zero_().to(self.device) for param in self.model.parameters()]

    def train(self, trans_learn: bool=False):
        self.model.to(self.device)
        self.model.train()

        # transfer learning stage
        # freeze all layers except the last one
        if trans_learn:
            params = self.model.parameters()
            self.model.fc.requires_grad = False
            # params_num = 0
            # for param in params:
            #     params_num += 1
            # for i, param in enumerate(params):
            #     # freeze all layers except the last one
            #     if i < params_num - 1:
            #         param.requires_grad = False
        
            train_params = filter(lambda p: p.requires_grad, params)
            if self.task_name == TaskName.CIFAR:
                self.optimizer = torch.optim.SGD(train_params, lr=self.lr, momentum=0.9, weight_decay=0.0001)
            elif self.task_name == TaskName.SPEECHCOMMAND:
                self.optimizer = torch.optim.Adam(train_params, lr=self.lr, weight_decay=0.0001)

        for i, param in enumerate(self.model.parameters()):
            self.temp_model_params[i] = param.detach().data

        # for i, c_d in enumerate(self.c_client):
        #     self.c_global[i] = c_d.clone().detach().data

        # reset loss and average gradient
        self.train_loss: float = 0
        for tensor in self.grad:
            tensor.zero_()

        # print(self.device)
        for i in range(self.local_epoch_num):
            for (X, label) in self.trainloader:
                # X.to(self.device)
                X = X.to(self.device)
                label = label.to(self.device)

                if self.task_name == TaskName.SPEECHCOMMAND:
                    self.transform = self.transform.to(self.device)

                    # X_ = X.clone()
                    # print(self.transform.device)
                    # print(X[0].shape)
                    X = self.transform(X)
                    # print(X[0].shape)
                    # X.to(self.device)
                    # X = X.to(self.device)

                y = self.model(X)
                if self.task_name == TaskName.SPEECHCOMMAND:
                    loss = self.loss_fn(y.squeeze(1), label)
                    # print(y.squeeze(1).shape)
                    # print(label.shape)
                else:
                    loss = self.loss_fn(y, label)

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
            
            # if self.task_name == TaskName.SPEECHCOMMAND:
            #     self.scheduler.step()


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
        
        # unfreeze all layers
        if trans_learn:
            self.model.fc.requires_grad = True
            if self.task_name == TaskName.CIFAR:
                self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9, weight_decay=0.0001) #
            elif self.task_name == TaskName.SPEECHCOMMAND:
                self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=0.0001)
        return self.train_loss, self.grad

    def set_lr(self, new_lr):
        if self.task_name == TaskName.CIFAR:
            self.lr = new_lr
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9, weight_decay=0.0001) #
        elif self.task_name == TaskName.SPEECHCOMMAND:
            self.lr = new_lr
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=0.0001)

    # def calc_training_cost(self, dataset_len: int) -> float:
    #     training_cost = 0.00495469 * dataset_len + 0.01023199
    #     return training_cost

class FedCLAR:

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
        self.fcaccu = open(self.config.result_dir + "caccu" + self.config.test_mark, "a")
        self.faccu.write("\nconfig:" + str(vars(self.config)) + "\n")
        self.floss.write("\nconfig:" + str(vars(self.config)) + "\n")
        self.fcost.write("\nconfig:" + str(vars(self.config)) + "\n")
        self.fcaccu.write("\nconfig:" + str(vars(self.config)) + "\n")
        self.faccu.flush()
        self.floss.flush()
        self.fcost.flush()
        self.fcaccu.flush()

        self.set_seed(data_seed)
        if self.config.task_name == TaskName.CIFAR:
            self.trainset, self.testset = load_dataset(self.config.task_name)
        elif self.config.task_name == TaskName.SPEECHCOMMAND:
            self.trainset, self.testset = TaskSpeechCommand.get_datasets(self.config)
            self.task = TaskSpeechCommand(self.config, testset=self.testset)
        partitioner = DatasetPartitioner(self.trainset, self.config.client_num, self.config.data_num_range, self.config.alpha, self.config.task_name)
        self.label_type_num = partitioner.label_type_num
        self.distributions = partitioner.get_distributions()
        self.partitioner: DatasetPartitioner = partitioner
        self.subsets_indices = partitioner.get_subsets()
        # partitioner.draw(20,"./pic/dubug.png")
        # global model, global drift in SCAFFOLD, clients
        self.model, self.c_global, self.clients = Client.init_clients(self.subsets_indices, self.config)
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
        pic_filename = self.config.result_dir + "group_distribution_" + self.config.test_mark + ".pdf"
        show_num = 10
        if show_num > len(self.groups):
            show_num = len(self.groups)
        self.inspect_group_distribution(self.groups, show_num, pic_filename)

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
                    if self.config.task_name == TaskName.CIFAR:
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
                            train_coefs = [0.07093414, -0.00559966 + 0.03344287872314453]
                        if self.config.train_method == Config.TrainMethod.FEDPROX:
                            train_coefs = [0.06719291, 0.14201339]
                    elif self.config.task_name == TaskName.SPEECHCOMMAND:
                        # audio
                        # distance coefficiences: [ 0.00079432 -0.00142096  0.02028448]
                        # training coefficiences: [0.00244381 0.10137752]
                        # fedprox coefficiences: [0.00247751 0.10650803]
                        # SCAFFOLD coefficiences: [0.00259606 0.10145685]
                        # sec agg: [0.00564634 0.04887151 0.03296802]
                        # scaffold agg: [ 0.01546567 -0.14052762  0.8393565 ]
                        group_coefs = [0.00564634, 0.04887151, 0.03296802]
                        # regular train coeffs
                        train_coefs = [0.00244381, 0.10137752]

                        if self.config.train_method == Config.TrainMethod.SCAFFOLD:
                            group_coefs = [ 0.01546567, -0.14052762,  0.8393565 ]
                            train_coefs = [0.00259606, 0.10145685]
                        if self.config.train_method == Config.TrainMethod.FEDPROX:
                            train_coefs = [0.00247751, 0.10650803]



                    training_cost = (train_coefs[0] * self.clients_data_nums[client_index] + train_coefs[1]) * self.config.local_epoch_num
                    group_overhead = (group_coefs[0] * (group_size*group_size) + group_coefs[1] * group_size + group_coefs[2])

                    client_cost = (training_cost + group_overhead) * self.config.group_epoch_num
                    group_cost += client_cost
                
                costs[i] = group_cost
            
            return costs

        def __FedCLAR_clustering(server_clients_arg: 'list[int]', server_no, th: float):
            def _sim_mat(model_sds: 'dict[str, Tensor]') -> np.ndarray:
                # models: 'list[nn.Module]' = [ self.clients[client_idx].model for client_idx in clients]
                # print(model_sds[0].keys())
                weights = [ sd['fc.weight'].detach().cpu().numpy().flatten() for sd in model_sds]
                biases = [ sd['fc.bias'].detach().cpu().numpy().flatten() for sd in model_sds]
                # concatenate corresponding weight and bias
                classifiers = [ np.concatenate((weights[i], biases[i])) for i in range(len(weights))]
                print('cluster number: ', len(model_sds))
                sim_mat = np.zeros((len(model_sds), len(model_sds)))
                for i in range(len(model_sds)):
                    for j in range(len(model_sds)):
                        # cosine similarity
                        sim_mat[i][j] = np.dot(classifiers[i], classifiers[j]) / (np.linalg.norm(classifiers[i]) * np.linalg.norm(classifiers[j]))
            
                return sim_mat

            def _merge_model_sds(state_dicts: 'list[dict[str, torch.Tensor]]', weights: 'list[float]') -> 'dict[str, torch.Tensor]':
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
                        avg_state_dict[key] = avg_state_dict[key].to(self.config.device)
                        state_dicts[i][key] = state_dicts[i][key].to(self.config.device)

                        avg_state_dict[key] += state_dicts[i][key] * weights[i]
                
                return avg_state_dict

            group_num_start = len(self.groups)
            clients = copy.deepcopy(server_clients_arg)
            clusters = [ [client_idx] for client_idx in clients ] # each client is a cluster
            model_sds = [ copy.deepcopy(self.clients[client_idx].model.state_dict()) for client_idx in clients]
            # sim_mat = _sim_mat(model_sds)

            while True:
                # update i, j
                sim_mat = _sim_mat(model_sds)
                for i in range(len(clusters)):
                    sim_mat[i][i] = -1
                min_index = np.argmax(sim_mat)
                i, j = min_index // len(clusters), min_index % len(clusters)
                lens = [ len(cluster) for cluster in clusters]
                max_len = max(lens)
                with open('min_sim.txt', 'a') as f:
                    f.write( str(sim_mat[i][j]) + ", " + str(len(clusters)) + ", " + str(max_len) + "\n")
                if len(clusters) < 2:
                    error_msg = "too few clusters! th=" + str(th) + ", cluster_num=" + str(len(clusters))
                    raise Exception(error_msg)
                # too hard to control the group size -_-
                # suspect the condition in the paper is 1 - similarity >= th
                # merge min_sim clusters tends to increase the next min_sim
                # then min_sim >= th does not work, cause min_sim is increasing
                # check min_sim.txt in the root folder
                # it records the min_sim, cluster_num and the max cluster size
                # conclusion: the result basically cannot be reproduced on CIFAR10
                # keep the original algorithm here, and add a term to control the group size
                # for comparison with Group-HFL

                # communicated with the authors and they confirmed that it should be argmax instead
                # have to mannually control the group size 
                # as th is very unstable
                if sim_mat[i, j] >= th and max_len <= self.config.min_group_size*2:
                    # set i < j
                    if i > j:
                        i, j = j, i
                    cluster_i = clusters.pop(i)
                    cluster_j = clusters.pop(j-1)
                    model_sd_i = model_sds.pop(i)
                    model_sd_j = model_sds.pop(j-1)

                    total_num = len(cluster_i) + len(cluster_j)
                    weights = [ len(cluster_i) / total_num, len(cluster_j) / total_num]
                    merged_sd = _merge_model_sds([model_sd_i, model_sd_j], weights)
                    merged_cluster = cluster_i + cluster_j

                    clusters.append(merged_cluster)
                    model_sds.append(merged_sd)
                else:
                    not_clustered_clients = []
                    for i, cluster in enumerate(clusters):
                        if len(cluster) == 1:
                            not_clustered_clients.append(cluster[0])
                    
                    # merge unclustered clients
                    clustered_model_sds = [ model_sd for i, model_sd in enumerate(model_sds) if len(clusters[i]) > 1]
                    clusters = [ cluster for i, cluster in enumerate(clusters) if len(cluster) > 1]
                    # send cluster models to clients
                    for cluster, model_sd in zip(clusters, clustered_model_sds):
                        for client_idx in cluster:
                            sd_copy = copy.deepcopy(model_sd)
                            self.clients[client_idx].model.load_state_dict(sd_copy)
                    # clusters.append(not_clustered_clients)
                    # merged_sd = _merge_model_sds([model_sds[i] for i in not_clustered_clients])
                    # clustered_model_sds.append(merged_sd)

                    # update groups
                    self.groups += clusters
                    self.groups_data_nums += [ sum([self.clients_data_nums[client_idx] for client_idx in cluster]) for cluster in clusters ]
                    self.groups_cvs += [ self.__calc_group_cv(cluster) for cluster in clusters ]
                    self.servers_groups[server_no] = [ group_num_start + i for i in range(len(clusters)) ]

                    break

        self.groups = [] 
        self.groups_cvs = []
        self.groups_data_nums = []

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
            elif self.config.grouping_mode == Config.GroupingMode.FEDCLAR:
                __FedCLAR_clustering(server_clients, i, self.config.FedCLAR_th)
            else:
                raise NotImplementedError

        self.groups_data_nums_arr = np.array(self.groups_data_nums)
        self.groups_weights_arr = self.groups_data_nums / np.sum(self.groups_data_nums)
        self.groups_cvs_arr = np.array(self.groups_cvs)
        self.groups_costs_arr = __calc_group_cost(self.groups)

        avg_group_size = np.mean(self.groups_data_nums_arr)
        sampling_data_num = self.config.sampling_frac * np.sum(self.clients_data_nums)
        self.sampling_num = round(sampling_data_num / avg_group_size)
        if self.sampling_num < 1:
            self.sampling_num = 1

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
            probs = np.full((len(self.groups), ), 1.0/len(self.groups), dtype=np.float32)
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
                state_dict = copy.deepcopy(self.clients[client_index].model.state_dict())
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
        selected_groups_models = []

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
            state_dict = copy.deepcopy(repr_client.model.state_dict())

            for key in state_dict_avg.keys():
                state_dict_avg[key] += state_dict[key] * weight

            # average delta drift in SCAFFOLD
            if self.config.train_method == Config.TrainMethod.SCAFFOLD:
                # delta drift in SCAFFOLD
                for i, c_d in enumerate(self.c_global):
                    self.c_global[i] += repr_client.c_client[i] * weight

        
        self.model.load_state_dict(state_dict_avg)

    def global_aggregate_by_clients(self, selected_groups: 'list[int]'):
        """
        only used for FedCLAR
        """

        # # init state dict
        client0 = self.clients[self.groups[selected_groups[0]][0]]
        state_dict_avg = copy.deepcopy(client0.model.state_dict()) 
        # self.c_global = [ c_d.clone().zero_().detach().data for c_d in client0.c_delta]
        for key in state_dict_avg.keys():
            state_dict_avg[key] = 0.0

        weights = self.clients_weights

        # get average state dict
        for group in selected_groups:
            for client_index in self.groups[group]:
                weight = weights[client_index]
                state_dict = copy.deepcopy(self.clients[client_index].model.state_dict())

                for key in state_dict_avg.keys():
                    state_dict_avg[key] += state_dict[key] * weight
        
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
        if self.config.task_name == TaskName.CIFAR:
            self.testloader = DataLoader(self.testset, 500, shuffle=True)
        elif self.config.task_name == TaskName.SPEECHCOMMAND:
            self.testloader = self.task.test_dataloader

        # print(self.groups)
        accus = []
        losses = []
        costs = []
        cluster_accus = []
        cur_cost = 0


        for i in range(self.config.global_epoch_num):
            if self.config.train_method == Config.TrainMethod.FEDCLAR:
                if i == self.config.FedCLAR_cluster_epoch:
                    self.global_distribute(range(len(self.groups)))
                    # clustering
                    # switch to FedCLAR
                    self.config.grouping_mode = Config.GroupingMode.FEDCLAR
                    # individual train for clustering
                    for epoch in range(self.config.group_epoch_num):
                        for client in self.clients:
                            client.train()
                    self.group()
                    # self.global_distribute(range(len(self.groups)))
                    print("FedCLAR clustering done")
                    print("groups number", len(self.groups))
                    # generate a test dataset for each cluster
                    # which has the same distribution as its training data
                    cluster_testloaders = []
                    for cluster_index in range(len(self.groups)):
                        cluster = self.groups[cluster_index]
                        distri_shape = len(self.distributions[0])
                        cluster_distri = np.zeros(distri_shape)
                        for client_index in cluster:
                            cluster_distri += self.distributions[client_index]
                        cluster_distri /= np.sum(cluster_distri)
                        # 100 test samples for each cluster
                        cluster_distri *= 100
                        cluster_distri = cluster_distri.astype(np.int32)
                        cluster_testset = self.partitioner.generate_new_dataset(cluster_distri)
                        cluster_testloader = DataLoader(cluster_testset, batch_size=100)
                        cluster_testloaders.append(cluster_testloader)

            # lr decay
            if i % self.config.lr_interval == self.config.lr_interval - 1:
                if self.config.task_name == TaskName.CIFAR:
                    for client in self.clients:
                        client.set_lr(client.lr / 10)
                    print('lr decay to {}'.format(self.clients[0].lr))
                elif self.config.task_name == TaskName.SPEECHCOMMAND:
                    for client in self.clients:
                        client.set_lr(client.lr / 5)

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

            if self.config.train_method == Config.TrainMethod.FEDCLAR:
                if i >= self.config.FedCLAR_cluster_epoch:
                    if i >= self.config.FedCLAR_tl_epoch:
                        # transfer learning training
                        # all devices train individually
                        for group_index in selected_groups:
                            if group_index == len(self.groups) - 1:
                                # only send global model to unclustered clients
                                self.global_distribute([group_index])
                            
                            # all clients in selected groups train individually
                            for i in range(self.config.group_epoch_num):
                                for client_index in self.groups[group_index]:
                                    self.clients[client_index].train(trans_learn=True)

                        # aggregate for global model
                        self.global_aggregate_by_clients(selected_groups)

                    else:
                        # clustered training
                        # # only distribute global model to the unclustered clients
                        # self.global_distribute([len(self.groups) - 1])
                        # clustered training and aggregation
                        self.global_train(range(len(self.groups)))
                        # self.global_train(selected_groups)
                        self.global_aggregate(range(len(self.groups)))
                        # self.global_aggregate(selected_groups)
                else:
                    # normal hierarchical training
                    self.global_distribute(selected_groups)
                    self.global_train(selected_groups)
                    self.global_aggregate(selected_groups)

            else:
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
            if i == self.config.FedCLAR_cluster_epoch:
                cur_cost += np.sum(self.groups_costs_arr)
            cur_cost += selected_cost


            # test and record
            if i % self.config.log_interval == self.config.log_interval - 1:
                if self.config.task_name == TaskName.CIFAR:
                    if i < self.config.FedCLAR_cluster_epoch:
                        cluster_accu = 0
                    else:
                        # test all selected clusters
                        avg_cluster_accu = 0
                        for cluster_index in selected_groups:
                            cluster_model = self.clients[self.groups[cluster_index][0]].model
                            cluster_testloader = cluster_testloaders[cluster_index]
                            cluster_accu, cluster_loss = test_model(cluster_model, cluster_testloader)
                            avg_cluster_accu += cluster_accu
                        cluster_accu = avg_cluster_accu / len(selected_groups)
                            
                        # cluster_accu, cluster_loss = test_model(cluster0_model, cluster_testloader)
                    accu, loss = test_model(self.model, self.testloader)
                elif self.config.task_name == TaskName.SPEECHCOMMAND:
                    self.task.model = self.model
                    accu, loss = self.task.test(self.testloader)
                    cluster_accu = 0

                self.faccu.write(f'{accu} ')
                self.floss.write(f'{loss} ')
                self.fcost.write(f'{cur_cost} ')
                self.fcaccu.write(f'{cluster_accu} ')
                self.faccu.flush()
                self.floss.flush()
                self.fcost.flush()
                self.fcaccu.flush()

                accus.append(accu)
                losses.append(loss)
                costs.append(cur_cost)
                cluster_accus.append(cluster_accu)

                quick_draw(accus, self.config.result_dir + 'accu' + str(self.config.test_mark) + '.png')
                quick_draw(cluster_accus, self.config.result_dir + 'caccu' + str(self.config.test_mark) + '.png')

                # print(f'epoch {i} accu: {accu} loss: {loss} cost: {cur_cost}')
                print(f'epoch {i} accu: {accu} loss: {loss} cost: {cur_cost} cluster accu: {cluster_accu}')
                
            if cur_cost > self.config.budget:
                break
        self.faccu.close()
        self.floss.close()
        self.fcost.close()
    
