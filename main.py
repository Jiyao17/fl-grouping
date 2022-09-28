
import torch

import numpy as np
import random
import copy

from utils.fed import GFL, Config
from utils.data import TaskName

base_config = Config(
    task_name=TaskName.CIFAR,
    server_num=10, client_num=1000, data_num_range=(10, 51), alpha=(0.1, 0.1),
    sampling_frac=0.2, budget=10**7,
    global_epoch_num=500,
    # the following line may vary
    group_epoch_num=5, local_epoch_num=1,
    lr=0.01, lr_interval=1000, local_batch_size=10,
    log_interval=5, 
    # the following two lines may vary
    grouping_mode=Config.GroupingMode.RANDOM, max_group_cv=0.1, min_group_size=10,
    selection_mode=Config.SelectionMode.RANDOM,
    device="cuda",
    train_method=Config.TrainMethod.SGD,
    data_path="./data/", 
    # the following 2 lines may vary
    result_dir="",
    test_mark="",
    comment="",
)

fedavg = copy.deepcopy(base_config)
fedavg.min_group_size = 1
fedavg.result_dir = "./exp_data/fedavg/"

grouping = copy.deepcopy(base_config)
# grouping.lr_interval = grouping.global_epoch_num // 2
grouping.group_epoch_num=10
grouping.local_epoch_num=2

cvg_cvs = copy.deepcopy(grouping)
cvg_cvs.grouping_mode = Config.GroupingMode.CV_GREEDY
cvg_cvs.selection_mode = Config.SelectionMode.PROB_RCV
cvg_cvs.result_dir = "./exp_data/grouping/cvg_cvs/"

cvg_rs = copy.deepcopy(grouping)
cvg_rs.grouping_mode = Config.GroupingMode.CV_GREEDY
cvg_rs.selection_mode = Config.SelectionMode.RANDOM
cvg_rs.result_dir = "./exp_data/grouping/cvg_rs/"

rg_cvs = copy.deepcopy(grouping)
rg_cvs.grouping_mode = Config.GroupingMode.RANDOM
rg_cvs.selection_mode = Config.SelectionMode.PROB_RCV
rg_cvs.result_dir = "./exp_data/grouping/rg_cvs/"

rg_rs = copy.deepcopy(grouping)
rg_rs.grouping_mode = Config.GroupingMode.RANDOM
rg_rs.selection_mode = Config.SelectionMode.RANDOM
rg_rs.result_dir = "./exp_data/grouping/rg_rs/"

var = copy.deepcopy(base_config)
var.grouping_mode = Config.GroupingMode.RANDOM
var.selection_mode = Config.SelectionMode.RANDOM
var.min_group_size = 15
var.result_dir = "./exp_data/grouping/var/"
# var.lr_interval = var.global_epoch_num // 2

var_10_100 = copy.deepcopy(var)
var_10_100.data_num_range = (10, 101)
var_10_100.test_mark = "_10-100"

var_10_50 = copy.deepcopy(var)
var_10_50.data_num_range = (10, 51)
var_10_50.test_mark = "_10-50"

var_lm = copy.deepcopy(var)
var_lm.data_num_range = (15, 46)
var_lm.test_mark = "_15-45"

var_20_40 = copy.deepcopy(var)
var_20_40.data_num_range = (20, 41)
var_20_40.test_mark = "_20-40"

var_medium = copy.deepcopy(var)
var_medium.data_num_range = (25, 36)
var_medium.test_mark = "_25-35"

var_30_30 = copy.deepcopy(var)
var_30_30.data_num_range = (30, 31)
var_30_30.test_mark = "_30-30"



debug = Config(
    task_name=TaskName.CIFAR,
    train_method=Config.TrainMethod.SGD,
    server_num=3, client_num=600, data_num_range=(10, 101), alpha=(0.1, 0.1),
    sampling_frac=0.2, budget=10**7,
    global_epoch_num=500, group_epoch_num=10, local_epoch_num=2,
    lr=0.01, lr_interval=1000, local_batch_size=10,
    log_interval=1, 
    # alpha=0.1: sigma = 
    grouping_mode=Config.GroupingMode.RANDOM, max_group_cv=0.1, min_group_size=10,
    # partition_mode=Config.PartitionMode.IID,
    selection_mode=Config.SelectionMode.RANDOM,
    device="cuda",
    data_path="./data/", 
    result_dir="./exp_data/debug/",
    test_mark="_sgd",
    comment="",
)

gs_comp = Config(
    task_name=TaskName.CIFAR,
    server_num=1, client_num=300, data_num_range=(20, 21), alpha=10,
    sampling_frac=0.2,
    global_epoch_num=1000,
    # the following line may vary
    group_epoch_num=1, local_epoch_num=5,
    lr=0.01, lr_interval=1000, local_batch_size=10,
    log_interval=5, 
    # the following two lines may vary
    grouping_mode=Config.GroupingMode.RANDOM, max_group_cv=0.1, min_group_size=10,
    selection_mode=Config.SelectionMode.RANDOM,
    aggregation_option=Config.AggregationOption.NUMERICAL_REGULARIZATION,
    device="cuda",
    data_path="./data/", 
    # the following 2 lines may vary
    result_dir="./exp_data/grouping/rg_rs/",
    test_mark="",
    comment="",

)



if __name__ == "__main__":

    # data_partition_seed = 0
    # seed = data_partition_seed
    # if seed is not None:
    #     random.seed(seed)
    #     np.random.seed(seed)
    #     torch.manual_seed(seed)
    #     torch.cuda.manual_seed(seed)
    # gs_comp.min_group_size = 50
    # gs_comp.test_mark = "_gs50"
    # config = gs_comp


    config = cvg_cvs
    config.aggregation_option = Config.AggregationOption.WEIGHTED_AVERAGE
    # config.grouping_mode = Config.GroupingMode.CV_GREEDY
    # config.selection_mode = Config.SelectionMode.PROB_RCV_COST
    config.min_group_size = 10
    config.max_group_cv = 1.0

    config.budget = 10**7
    config.train_method = Config.TrainMethod.SGD
    config.log_interval = 5

    config.group_epoch_num = 5
    config.local_epoch_num = 1
    config.alpha = (0.1, 0.1)


    # 0.05 0.1 0.5 0.01

    # config.selection_mode = Config.SelectionMode.PROB_RCV_COST
    config.test_mark = "_alpha0.1_cv1.0_5*1"
    config.comment = "weighted average only"
    gfl = GFL(config)
    gfl.run()






