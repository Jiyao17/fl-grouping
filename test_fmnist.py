
import copy

from utils.fed import Config, TaskName, GFL
from utils.model import FMNIST

fmnist_base = Config(
    task_name=TaskName.FMNIST,
    server_num=1, client_num=100, data_num_range=(20, 201), alpha=(0.1, 0.1),
    sampling_frac=0.2, budget=10**7,
    global_epoch_num=1000, 
    # the following line may vary
    group_epoch_num=5, local_epoch_num=2,
    lr=0.001, lr_interval=1000, local_batch_size=10,
    log_interval=1, 
    # the following two lines may vary
    grouping_mode=Config.GroupingMode.RANDOM, max_group_cv=1.0, min_group_size=5,
    selection_mode=Config.SelectionMode.RANDOM,
    aggregation_option=Config.AggregationOption.WEIGHTED_AVERAGE,
    device="cuda",
    train_method=Config.TrainMethod.SGD,
    data_path="./data/", 
    # the following 2 lines may vary
    result_dir="./exp_data/grouping/rg_rs/", 
    test_mark="_fmnist",
    comment="",
)

# fmnist_base.device = "cuda:3"
# gfl = GFL(fmnist_base)
# gfl.run()

model = FMNIST()
print(model)
sd = model.state_dict()
print(sd.keys())