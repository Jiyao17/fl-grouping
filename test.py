

from utils.data import dataset_split_r_random, subsets_to_target_set
from utils.task import TaskCIFAR, ExpConfig
from utils.hierarchy import Client

exp_config = ExpConfig("iid", group_epoch_num=500, local_epoch_num=5,
                group_size=100, result_dir="./cifar/iid/", simulation_num=1)
trainset, testset = TaskCIFAR.load_dataset("./data/")
subsets = dataset_split_r_random(trainset, 10, 5000, 5)

# print(len(subsets[2]))



# trainset, testset = TaskCIFAR.load_dataset("./data/")

# subsets = dataset_split_r(trainset, 100, 100, 3)

lable_list = []
for (sample, lable) in subsets[2]:
    lable_list.append(lable)

print(lable_list)

# sum = 0
# for i in range(10):
#     count = lable_list.count(i)
#     print(count)
#     sum += count

s = subsets_to_target_set(subsets[0:3])

print(s)