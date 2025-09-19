
import copy
from multiprocessing import Process

from utils.fed import Config, TaskName, GFL

config_100 = Config(
    task_name=TaskName.CIFAR,
    server_num=1, client_num=100, data_num_range=(20, 201), alpha=(0.1, 0.1),
    sampling_frac=0.2, budget=10**7,
    global_epoch_num=1000, 
    # the following line may vary
    group_epoch_num=5, local_epoch_num=2,
    lr=0.01, lr_interval=1000, local_batch_size=10,
    log_interval=1, 
    # the following two lines may vary
    grouping_mode=Config.GroupingMode.QCID, max_group_cv=1.0, min_group_size=5,
    regroup_interval=1,
    selection_mode=Config.SelectionMode.RANDOM,
    aggregation_option=Config.AggregationOption.WEIGHTED_AVERAGE,
    device="cuda:0",
    train_method=Config.TrainMethod.SGD,
    data_path="./data/", 
    # the following 2 lines may vary
    result_dir="./exp_data/grouping/qcid/", 
    test_mark="_100",
    comment="",
)

config_200 = copy.deepcopy(config_100)
config_200.client_num = 200
config_200.server_num = 2
config_200.test_mark = "_200"
config_200.device = "cuda:0"


config_300 = copy.deepcopy(config_100)
config_300.client_num = 300
config_300.server_num = 3
config_300.test_mark = "_300"
config_300.device = "cuda:0"

def process_run(config: Config):
    gfl = GFL(config)
    gfl.run()

if __name__ == "__main__":
    configs = [config_100, config_200, config_300]
    for config in configs:
        p = Process(target=process_run, args=(config,))
        p.start()
