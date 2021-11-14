
# code source:
# FashionMNIST: pytorch tutorial quickstart
# SpeechCommand: pytorch tutorial audio
# TextClassification: pytorch tutorial text




from utils.task import ExpConfig
from utils.simulator import Simulator


# def global_run():

#     # create hierarchical structure
#     trainset, testset = TaskCIFAR.load_dataset("./data/")
#     subsets = dataset_split_r(trainset, client_num, data_num, r)
#     config = TrainConfig(local_epoch_num, batch_size, lr, device)

#     clients = [ Client(TaskCIFAR(subsets[i], config))
#         for i in range(client_num) ]
#     counter = 0
#     groups = []
#     for i in range(group_num):
#         group_clients = []
#         for j in range(group_size):
#             group_clients.append(clients[counter])
#             counter += 1
#         group = Group(group_clients, local_epoch_num)
#         groups.append(group)
#     global_sys = Global(groups, group_epoch_num)

    
#     _, testset = TaskCIFAR.load_dataset("./data/")

#     dataloader: DataLoader = DataLoader(testset, batch_size=256,
#         shuffle=False)

#     for i in range(global_epoch_num):
#         global_sys.round()

#         acc, loss = global_sys.test_model(dataloader, device)

#         print("Epoch %d: accuracy=%.2f" % (i, acc))


# def group_run():
#     # create hierarchical structure
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     trainset, testset = TaskCIFAR.load_dataset("./data/")
#     subsets = dataset_split_r(trainset, client_num, data_num, r)
#     config = TrainConfig(local_epoch_num, batch_size, lr, device)

#     clients = [ Client(TaskCIFAR(subsets[i], config))
#         for i in range(client_num) ]
#     counter = 0
#     group_clients: list[Client] = []
#     for j in range(group_size):
#         group_clients.append(clients[counter])
#         counter += 1
#     group = Group(group_clients, local_epoch_num)

#     dataloader: DataLoader = DataLoader(testset, batch_size=256,
#         shuffle=False)

#     file = open("")
#     for i in range(group_epoch_num):
#         sd = deepcopy(group.model.state_dict())
#         group.distribute_model()
#         group.train_model()
#         group.aggregate_model()
#         sd1 = group.model.state_dict()

#         acc, loss = TaskCIFAR.test_model(group_clients[0].get_model(), dataloader, device)

#         print("Epoch %d: accuracy=%.2f" % (i, acc))


if __name__ == "__main__":
    # prepare for multi-proccessing
    # if get_start_method(False) != "spawn":
    # set_start_method("spawn")


    # default test config for iid
    exp_config = ExpConfig(test_type="test", group_epoch_num=1000,
        local_epoch_num=1, client_num=1, group_size=1, group_num=1,
        local_data_num=50000, batch_size=50, lr=0.001, noniid_degree=10,
        simulation_num=1, result_dir="./cifar/test/")

    # default experimental config for iid
    # exp_config = ExpConfig("noniid", group_epoch_num=500, local_epoch_num=5, client_num=25,
    #                 group_size=25, group_num=1, local_data_num=2000, batch_size= 50, 
    #                 noniid_degree=5, lr=0.01,
    #                 result_dir="./cifar/noniid/", simulation_num=3)

    # exp_config = ExpConfig()
    # exp_config = ExpConfig("noniid", group_epoch_num=1000, local_epoch_num=5, client_num=25,
    #                 group_size=25, group_num=1, local_data_num=2000, batch_size= 50, 
    #                 noniid_degree=10, lr=0.01,
    #                 result_dir="./cifar/iid/", simulation_num=3)

    sim = Simulator(exp_config)
    sim.start()
