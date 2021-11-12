
# code source:
# FashionMNIST: pytorch tutorial quickstart
# SpeechCommand: pytorch tutorial audio
# TextClassification: pytorch tutorial text

import json
from multiprocessing import set_start_method

from utils.simulator import Simulator
from utils.tasks import Config, UniTask



if __name__ == "__main__":
    # prepare for multi-proccessing
    # if get_start_method(False) != "spawn":
    set_start_method("spawn")

    sigma = [9, 8, 5, 3, 2]

    config = Config()
    config.task_name = UniTask.supported_tasks[0]
    config.client_num = 100
    config.l_data_num = 600
    config.l_epoch_num = 5
    config.l_batch_size = 10
    config.g_epoch_num = 500
    config.sigma = sigma
    config.simulation_num = 3
    
    config.result_dir = "./result-noniid/"
    config.test_type = "non-iid-grouping"

    simulator: Simulator = Simulator()


    simulator.start()

    




