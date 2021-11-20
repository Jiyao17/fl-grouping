
from argparse import ArgumentParser
from copy import deepcopy
from io import TextIOWrapper
from multiprocessing import Process
from torch.utils.data import DataLoader, Subset


from utils.task import ExpConfig, TaskCIFAR
from utils.data import dataset_split_r, dataset_split_r_random_distinct, grouping, regroup
from utils.hierarchy import Client, Group, Global



def single_simulation(configs: ExpConfig):
    ssimulator = __SingleSimulator(configs)
    ssimulator.start()

class __SingleSimulator:
    def __init__(self, config: ExpConfig) -> None:
        self.config = config

    def start(self):
        accu_file: str = self.config.result_dir + "accu" + str(self.config.simulation_index)
        loss_file: str = self.config.result_dir + "loss" + str(self.config.simulation_index)
        faccu = open(accu_file, "a")
        floss = open(loss_file, "a")
        # write experiment settings
        faccu.write(str(vars(self.config)) + "\n")
        faccu.flush()
        floss.write(str(vars(self.config)) + "\n")
        floss.flush()

        if self.config.task_type == "test":
            self.run_exp_test(faccu, floss)
        elif self.config.task_type == "r":
            self.run_exp_noniid_r(faccu, floss)
        elif self.config.task_type == "grouping":
            self.run_exp_grouping(faccu, floss)

        faccu.write("\n")
        floss.write("\n")
        faccu.close()
        floss.close()

    def run_exp_test(self, faccu: TextIOWrapper, floss: TextIOWrapper):
        TCLASS = self.config.get_task_class()
        # create hierarchical structure
        trainset, testset = TCLASS.load_dataset(self.config.datapath)
        testloader = DataLoader(testset, 500)

        trainer = TCLASS(trainset, self.config)
        for i in range(self.config.group_epoch_num):
            # print("epoch %d" % (i,))
            trainer.train_model()
            
            if i % self.config.log_interval == self.config.log_interval - 1:
                accu, loss = TCLASS.test_model(trainer.model, testloader, self.config.device)
                faccu.write("{:.5f} ".format(accu))
                faccu.flush()
                floss.write("{:.5f} ".format(loss))
                floss.flush()

    def run_exp_noniid_r(self, faccu: TextIOWrapper, floss: TextIOWrapper):
        TCLASS = self.config.get_task_class()
        MCLASS = self.config.get_model_class()
        # create hierarchical structure
        trainset, testset = TCLASS.load_dataset(self.config.datapath)
        subsets = dataset_split_r(trainset, self.config.client_num,
            self.config.local_data_num, self.config.noniid_degree)
        testloader = DataLoader(testset, 500)

        clients = [ Client(TCLASS(subsets[i], self.config), self.config)
            for i in range(self.config.client_num) ]

        group = Group(clients, self.config, MCLASS())
        for i in range(self.config.group_epoch_num):
            group.round()

            if i % self.config.log_interval == self.config.log_interval - 1:
                accu, loss = TCLASS.test_model(group.model, testloader, self.config.device)
                faccu.write("{:.5f} ".format(accu))
                faccu.flush()
                floss.write("{:.5f} ".format(loss))
                floss.flush()

    def run_exp_grouping(self, faccu: TextIOWrapper, floss: TextIOWrapper):
        TCLASS = self.config.get_task_class()
        MCLASS = self.config.get_model_class()
        # create hierarchical structure
        trainset, testset = TCLASS.load_dataset(self.config.datapath)
        indices_list = dataset_split_r_random_distinct(trainset, 500, 5)
        groups = grouping(indices_list, trainset.targets, 10)
        total_data_num = 0
        for group in groups:
            total_data_num += len(group)

        # actually indices for groups
        groups = regroup(groups, self.config.group_num)
        # real groups of clients
        real_groups = []
        for group in groups:
            
            new_group = [ Client(TCLASS(Subset(trainset, client_data), self.config),
                                    self.config) 
                            for client_data in group ]

            # form groups
            group = Group(new_group, self.config, MCLASS())
            real_groups.append(group)

        # output real configs
        faccu.write("Total data num = {:.d}, group num = {:.d} \n".format(total_data_num, len(real_groups)))
        faccu.flush()
        floss.write("Total data num = {:.d}, group num = {:.d} \n".format(total_data_num, len(real_groups)))
        floss.flush()

        # form global
        global_sys = Global(groups, self.config, MCLASS())
        testloader: DataLoader = DataLoader(testset, 500)
        for i in range(self.config.global_epoch_num):
            global_sys.round()

            if i % self.config.log_interval == self.config.log_interval - 1:
                accu, loss = TCLASS.test_model(global_sys.get_model(),
                                        testloader, self.config.device)
                faccu.write("{:.5f} ".format(accu))
                faccu.flush()
                floss.write("{:.5f} ".format(loss))
                floss.flush()


class Simulator:

    def __init__(self, config: ExpConfig = None) -> None:
        self.config = deepcopy(config)

    def start(self):
        print(vars(self.config))

        procs: list[Process] = []
        for i in range(self.config.simulation_num):
            self.config.simulation_index = i
            proc = Process(target=single_simulation, args=(self.config, ))
            proc.start()
            procs.append(proc)

        for proc in procs:
            proc.join()

    def set_configs(self, config: ExpConfig):
        self.config = deepcopy(config)


