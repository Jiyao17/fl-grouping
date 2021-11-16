
# code source:
# FashionMNIST: pytorch tutorial quickstart
# SpeechCommand: pytorch tutorial audio
# TextClassification: pytorch tutorial text

from multiprocessing import set_start_method


from utils.task import ExpConfig
from utils.simulator import Simulator




if __name__ == "__main__":
    # prepare for multi-proccessing
    # if get_start_method(False) != "spawn":
    set_start_method("spawn")


    # baseline: central
    # exp_config = ExpConfig(test_type="test", group_epoch_num=200,
    #     local_epoch_num=1, client_num=1, group_size=1, group_num=1,
    #     local_data_num=50000, batch_size=50, lr=0.003, noniid_degree=10,
    #     simulation_num=3, result_dir="./cifar/temp/",
    #     log_interval=5,
    #     comment="central baseline, decay lr")

    # default experimental config for iid
    exp_config = ExpConfig("noniid", group_epoch_num=200,
                    local_epoch_num=5, client_num=100, group_size=100, group_num=1,
                    local_data_num=500, batch_size= 50, lr=0.003, noniid_degree=5, 
                    simulation_num=3, result_dir="./cifar/noniid/", 
                    comment="iid baseline of FedAvg",
                    )

    # exp_config = ExpConfig()
    # exp_config = ExpConfig("noniid", group_epoch_num=1000, local_epoch_num=5, client_num=25,
    #                 group_size=25, group_num=1, local_data_num=2000, batch_size= 50, 
    #                 noniid_degree=10, lr=0.01,
    #                 result_dir="./cifar/iid/", simulation_num=3)

    sim = Simulator(exp_config)
    sim.start()
