
import torch

import numpy as np
import random
import copy
from multiprocessing import Process

from utils.fed import GFL, Config
from utils.data import TaskName

base_config = Config(
    task_name=TaskName.CIFAR,
    server_num=3, client_num=300, data_num_range=(20, 201), alpha=(0.1, 0.1),
    sampling_frac=0.2, budget=10**8,
    global_epoch_num=1000, 
    # the following line may vary
    group_epoch_num=10, local_epoch_num=2,
    lr=0.01, lr_interval=1000, local_batch_size=10,
    log_interval=5, 
    # the following two lines may vary
    grouping_mode=Config.GroupingMode.RANDOM, max_group_cv=0.5, min_group_size=10,
    selection_mode=Config.SelectionMode.RANDOM,
    aggregation_option=Config.AggregationOption.WEIGHTED_AVERAGE,
    device="cuda",
    train_method=Config.TrainMethod.SGD,
    data_path="./data/", 
    # the following 2 lines may vary
    result_dir="./exp_data/grouping/rg_rs/", 
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
cvg_cvs.selection_mode = Config.SelectionMode.PROB_ESRCV
cvg_cvs.result_dir = "./exp_data/grouping/cvg_cvs/"

cvg_rs = copy.deepcopy(grouping)
cvg_rs.grouping_mode = Config.GroupingMode.CV_GREEDY
cvg_rs.selection_mode = Config.SelectionMode.RANDOM
cvg_rs.result_dir = "./exp_data/grouping/cvg_rs/"

rg_cvs = copy.deepcopy(grouping)
rg_cvs.grouping_mode = Config.GroupingMode.RANDOM
rg_cvs.selection_mode = Config.SelectionMode.PROB_ESRCV
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
    server_num=3, client_num=300, data_num_range=(20, 201), alpha=(0.1, 0.1),
    sampling_frac=0.2, budget=10**7,
    global_epoch_num=100, group_epoch_num=10, local_epoch_num=2,
    lr=0.01, lr_interval=1000, local_batch_size=10,
    log_interval=1, 
    # alpha=0.1: sigma = 
    grouping_mode=Config.GroupingMode.CV_GREEDY, max_group_cv=1, min_group_size=5,
    # partition_mode=Config.PartitionMode.IID,
    selection_mode=Config.SelectionMode.PROB_SRCV,
    device="cuda",
    data_path="./data/", 
    result_dir="./exp_data/debug/",
    test_mark="_sc",
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
    aggregation_option=Config.AggregationOption.WEIGHTED_AVERAGE,
    device="cuda",
    data_path="./data/",
    # the following 2 lines may vary
    result_dir="./exp_data/grouping/rg_rs/",
    test_mark="",
    comment="",

)


comp_base = copy.deepcopy(base_config)
comp_base.server_num = 3
comp_base.client_num = 300
comp_base.alpha = (0.1, 0.1)
comp_base.max_group_cv = 1.0
comp_base.min_group_size = 5
comp_base.data_num_range = (20, 201)
comp_base.group_epoch_num = 5
comp_base.local_epoch_num = 2
comp_base.log_interval = 5
comp_base.budget = 1.e7

# FedCLAR = copy.deepcopy(comp_base)
# FedCLAR.FedCLAR_cluster_epoch = 50
# FedCLAR.FedCLAR_tl_epoch = 200
# FedCLAR.train_method = Config.TrainMethod.FEDCLAR
# FedCLAR.grouping_mode = Config.GroupingMode.RANDOM # switch to FEDCLAR at given epoch
# FedCLAR.result_dir = "./exp_data/grouping/fedclar/"

gamma = copy.deepcopy(comp_base)
gamma.server_num = 1
gamma.client_num = 100
gamma.data_num_range = (20, 201)
gamma.test_mark = "_gamma"

gamma_debug = copy.deepcopy(gamma)
gamma_debug.data_num_range = (110, 111)
gamma_debug.test_mark = "_gamma_debug"


FedProx = copy.deepcopy(comp_base)
FedProx.train_method = Config.TrainMethod.FEDPROX
FedProx.result_dir = "./exp_data/grouping/rg_rs/fedprox/"

scaffold = copy.deepcopy(comp_base)
scaffold.train_method = Config.TrainMethod.SCAFFOLD
scaffold.result_dir = "./exp_data/grouping/rg_rs/scaffold/"

comp_cvg_cvs = copy.deepcopy(comp_base)
comp_cvg_cvs.grouping_mode = Config.GroupingMode.CV_GREEDY
comp_cvg_cvs.selection_mode = Config.SelectionMode.PROB_ESRCV
comp_cvg_cvs.result_dir = "./exp_data/grouping/cvg_cvs/"

scaffold_cvg_cvs = copy.deepcopy(comp_cvg_cvs)
scaffold_cvg_cvs.train_method = Config.TrainMethod.SCAFFOLD
scaffold_cvg_cvs.result_dir = "./exp_data/grouping/cvg_cvs/scaffold/"

fedprox_cvg_cvs = copy.deepcopy(comp_cvg_cvs)
fedprox_cvg_cvs.train_method = Config.TrainMethod.FEDPROX
fedprox_cvg_cvs.result_dir = "./exp_data/grouping/cvg_cvs/fedprox/"

audio_configs = [comp_base, FedProx, scaffold, comp_cvg_cvs, fedprox_cvg_cvs, scaffold_cvg_cvs, ]

for i, config in enumerate(audio_configs):
    audio_configs[i] = copy.deepcopy(config)
    audio_configs[i].task_name = TaskName.SPEECHCOMMAND
    audio_configs[i].server_num = 1
    audio_configs[i].client_num = 150
    audio_configs[i].data_num_range = (180, 181)
    audio_configs[i].sampling_frac = 0.2
    # cv=1.0 gs=5
    # 0.5 10
    audio_configs[i].global_epoch_num = 1000

    audio_configs[i].group_epoch_num = 5
    audio_configs[i].local_epoch_num = 2
    audio_configs[i].alpha = (0.01, 0.01)
    audio_configs[i].max_group_cv = 10.0
    audio_configs[i].min_group_size = 15
    audio_configs[i].lr = 0.01
    audio_configs[i].lr_interval = 50
    audio_configs[i].log_interval = 5

    audio_configs[i].test_mark = "_sc"

def process_run(config: Config):
    gfl = GFL(config)
    gfl.run()


ouea = copy.deepcopy(comp_base)
ouea.grouping_mode = Config.GroupingMode.OUEA
ouea.result_dir = "./exp_data/grouping/ouea/"

ouea_sc = copy.deepcopy(audio_configs[0])
ouea_sc.grouping_mode = Config.GroupingMode.OUEA
# ouea_sc.data_num_range = (50, 501)
ouea_sc.result_dir = "./exp_data/grouping/ouea/"


ouea_debug = copy.deepcopy(ouea)
# ouea_debug.server_num = 1
# ouea_debug.client_num = 100
ouea_debug.test_mark += "_debug"

kld = copy.deepcopy(comp_base)
kld.grouping_mode = Config.GroupingMode.KLD
kld.selection_mode = Config.SelectionMode.RANDOM
kld.result_dir = "./exp_data/grouping/kld/"

kld_sc = copy.deepcopy(audio_configs[0])
kld_sc.grouping_mode = Config.GroupingMode.KLD
# kld_sc.data_num_range = (50, 501)
kld_sc.result_dir = "./exp_data/grouping/kld/"

kld_cvs = copy.deepcopy(kld)
kld_cvs.selection_mode = Config.SelectionMode.PROB_ESRCV

kld_debug = copy.deepcopy(kld)
# kld_debug.server_num = 1
# kld_debug.client_num = 100
kld_debug.test_mark += "_debug"

rg_sc = copy.deepcopy(kld_sc)
rg_sc.grouping_mode = Config.GroupingMode.RANDOM
# rg_sc.data_num_range = (50, 501)
rg_sc.result_dir = "./exp_data/grouping/rg_rs/"

cvg_sc = copy.deepcopy(kld_sc)
cvg_sc.grouping_mode = Config.GroupingMode.CV_GREEDY
# cvg_sc.data_num_range = (50, 501)
cvg_sc.result_dir = "./exp_data/grouping/cvg_cvs/"
cvg_cvs.test_mark = "_grouping"

if __name__ == "__main__":
    # gfl = GFL(debug)
    # gfl.run()

    # exit()

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
    CUDAS = [2, 3, 6, 7]
    configs = [comp_base, FedProx, scaffold, comp_cvg_cvs, fedprox_cvg_cvs, scaffold_cvg_cvs, ouea, kld]
    # configs = [comp_base, FedProx, scaffold, ouea, kld, comp_cvg_cvs,]
    # configs = [configs[0], configs[3]]
    # configs = [configs[1], configs[4]]
    # configs = [configs[2], configs[5]]
    configs = [configs[3]]
    # configs[0].test_mark += "_test"


    # configs = [comp_cvg_cvs]
    # configs[0].grouping_mode = Config.GroupingMode.OUEA
    # configs[0].test_mark = "grouping"

    # configs = [audio_configs[0], audio_configs[3]]
    # configs = [audio_configs[1], audio_configs[4]]
    # configs = [audio_configs[2], audio_configs[5]]
    configs = [audio_configs[3]]

    # configs = [ouea_debug]
    # configs = [ouea_sc]
    # configs = [kld_debug]
    # kld.selection_mode = Config.SelectionMode.PROB_ESRCV
    # kld.test_mark = "_esrcv"
    # configs = [kld]
    # configs = [kld_sc]
    # configs = [kld_debug]
    # configs = [gamma]
    # configs = [gamma_debug]
    # configs = [rg_sc]
    # configs = [cvg_sc]

    task_counter = 0
    for i, config in enumerate(configs):

        # config.group_epoch_num = 5
        # config.local_epoch_num = 2
        # config.log_interval = 5
        
        cvg_cvs_mark_base = "_alpha" + str(config.alpha[1]) + "_cv" + str(config.max_group_cv) + "_" \
            + str(config.group_epoch_num) + "*" + str(config.local_epoch_num)
        rg_rs_mark_base = "_alpha" + str(config.alpha[1]) + "_gs" + str(config.min_group_size) + "_" \
            + str(config.group_epoch_num) + "*" + str(config.local_epoch_num)
        # if i < 1:
        config.test_mark += rg_rs_mark_base
        # else:
        # config.test_mark += cvg_cvs_mark_base

        # config.test_mark += "cv_dn"
        # p = Process(target=process_run, args=(config,))
        # task_counter += 1
        # p.start()
        gfl = GFL(config)
        gfl.run()



    

    # config.alpha = (0.1, 0.1)
    # config.server_num = 1
    # config.client_num = 100
    # config.data_num_range = (20, 101)
    # config.aggregation_option = Config.AggregationOption.NUMERICAL_REGULARIZATION
    # config.grouping_mode = Config.GroupingMode.CV_GREEDY
    # config.selection_mode = Config.SelectionMode.PROB_ESRCV

    # config.min_group_size = 20
    # config.max_group_cv = 1.0
    # config.group_epoch_num = 10
    # config.local_epoch_num = 2

    # config.budget = 10**8
    # config.train_method = Config.TrainMethod.SGD
    # config.log_interval = 5




    # 0.05 0.1 0.5 0.01

    # config.selection_mode = Config.SelectionMode.PROB_RCV_COST
    # config.test_mark = "_alpha0.1_cv1.0_5*1"

    # config.test_mark = rg_rs_mark_base
    # config.comment = "weighted average only"






