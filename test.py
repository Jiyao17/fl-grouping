

from torch.utils.data.dataset import Subset
from utils.data import dataset_split_r_random, dataset_split_r_random_distinct, datasets_to_target_sets, grouping
from utils.task import TaskCIFAR, ExpConfig
from utils.hierarchy import Client

exp_config = ExpConfig("iid", group_epoch_num=500, local_epoch_num=5,
                group_size=100, result_dir="./cifar/iid/", simulation_num=1)
trainset, testset = TaskCIFAR.load_dataset("./data/")
indices_list = dataset_split_r_random_distinct(trainset, 100, 5)
print("client num: %d " % (len(indices_list),))
print(len(indices_list[-1]))

groups = grouping(indices_list, trainset.targets, 10)
total_num = sum([len(group) for group in groups])
print("group num: %d " % (len(groups),))
print(total_num)
# print(len(subsets[2]))


# trainset, testset = TaskCIFAR.load_dataset("./data/")

# subsets = dataset_split_r(trainset, 100, 100, 3)
for group in groups:
    subset = Subset(trainset, group)
    lable_list = []
    for (sample, lable) in subset:
        lable_list.append(lable)

    # print(lable_list)

    sum = 0
    for i in range(10):
        count = lable_list.count(i)
        print(count, end=" ")
        sum += count
    print(sum)
# s = subsets_to_target_set(subsets[0:3])

# print(s)