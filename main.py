
import torch

import numpy as np
import random
import copy

from utils.fed import GFL, Config
from utils.data import TaskName

base_config = Config(
    task_name=TaskName.CIFAR,
    server_num=5, client_num=500, data_num_range=(50, 101), alpha=0.1,
    sampling_frac=0.2, budget=10**7,
    global_epoch_num=1000, 
    # the following line may vary
    group_epoch_num=5, local_epoch_num=1,
    lr=0.01, lr_interval=1000, local_batch_size=10,
    log_interval=5, 
    # the following two lines may vary 
    grouping_mode=Config.GroupingMode.CV_GREEDY, max_group_cv=0.1, min_group_size=10,
    selection_mode=Config.SelectionMode.PROB_RCV,
    device="cuda",
    data_path="./data/", 
    # the following 2 lines may vary
    result_dir="",
    test_mark="",
    comment="",

)

fedavg = copy.deepcopy(base_config)
fedavg.result_dir = "./exp_data/fedavg/"

cvg_cvs = copy.deepcopy(base_config)
cvg_cvs.group_epoch_num=5
cvg_cvs.local_epoch_num=1
cvg_cvs.min_group_size=10
cvg_cvs.grouping_mode = Config.GroupingMode.CV_GREEDY
cvg_cvs.selection_mode = Config.SelectionMode.PROB_ESRCV
cvg_cvs.result_dir = "./exp_data/grouping/cvg_cvs/"

cvg_rs = copy.deepcopy(base_config)
cvg_rs.group_epoch_num=5
cvg_rs.local_epoch_num=1
cvg_rs.grouping_mode = Config.GroupingMode.CV_GREEDY
cvg_rs.selection_mode = Config.SelectionMode.RANDOM
cvg_rs.result_dir = "./exp_data/grouping/cvg_rs/"

rg_cvs = copy.deepcopy(base_config)
rg_cvs.group_epoch_num=5
rg_cvs.local_epoch_num=1
rg_cvs.grouping_mode = Config.GroupingMode.RANDOM
rg_cvs.selection_mode = Config.SelectionMode.PROB_ESRCV
rg_cvs.result_dir = "./exp_data/grouping/rg_cvs/"

rg_rs = copy.deepcopy(base_config)
rg_rs.group_epoch_num=5
rg_rs.local_epoch_num=1
rg_rs.grouping_mode = Config.GroupingMode.RANDOM
rg_rs.selection_mode = Config.SelectionMode.RANDOM
rg_rs.result_dir = "./exp_data/grouping/rg_rs/"


debug = Config(
    task_name=TaskName.CIFAR,
    server_num=10, client_num=1000, data_num_range=(10, 50), alpha=0.1,
    sampling_frac=0.2,
    global_epoch_num=300, group_epoch_num=5, local_epoch_num=1,
    lr=0.1, lr_interval=1000, local_batch_size=10,
    log_interval=5, 
    # alpha=0.1: sigma = 
    grouping_mode=Config.GroupingMode.CV_GREEDY, max_group_cv=0.1, min_group_size=10,
    # partition_mode=Config.PartitionMode.IID,
    selection_mode=Config.SelectionMode.PROB_ESRCV,
    device="cuda",
    data_path="./data/", 
    result_dir="./exp_data/debug/",
    test_mark="",
    comment="",
)

def gs_comp():
    gses = [5, 10, 15, 20, 25]
    for gs in gses:
        config = copy.deepcopy(rg_rs)
        config.min_group_size = gs
        config.test_mark = "a0.1_gs" + str(gs) + "_5*1"
        GFL(config).run()

def cv_comp():
    cvs = [0.01, 0.1, 1.0]
    for cv in cvs:
        config = copy.deepcopy(cvg_cvs)
        config.max_group_cv = cv
        config.test_mark = "a0.1_cv" + str(cv) + "_5*1"
        GFL(config).run()


if __name__ == "__main__":

    # seed = None
    # if seed is not None:
    #     random.seed(seed)
    #     np.random.seed(seed)
    #     torch.manual_seed(seed)
    #     torch.cuda.manual_seed(seed)

    config = rg_rs
    config.max_group_cv = 0.05
    config.min_group_size = 10
    # config.test_mark = "a0.1_cv0.05_5*1"
    config.test_mark = "a0.1_gs10_5*1"
    config.comment = "5, 500, 50-100"
    gfl = GFL(config)
    gfl.run()




