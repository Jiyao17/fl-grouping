

from torch.utils.data.dataset import Subset
from utils.data import dataset_split_r_random, load_dataset

trainset, testset = load_dataset("./data/")
indices_list = dataset_split_r_random(trainset, 2500, 20, 5)
# print("client num: %d " % (len(indices_list),))
# print(len(indices_list[-1]))

# groups = grouping(indices_list, trainset.targets, 10)

# total_client_num = 0
# total_data_num = 0
# for group in groups:
#     total_client_num += len(group)
#     for client in group:
#         total_data_num += len(client)

# print("total client num = %d, total data num = %d" % (total_client_num, total_data_num))

# # print(len(subsets[2]))
# groups = regroup(groups, 10)

# trainset, testset = TaskCIFAR.load_dataset("./data/")

# subsets = dataset_split_r(trainset, 100, 100, 3)

client_lable_list = []
client_indices = indices_list[0]
subset = Subset(trainset, client_indices)
    
for (sample, lable) in subset:
    client_lable_list.append(lable)

print(client_lable_list)


# for indices in indices_list:
#     group_lable_list = []
#     for client_data in group:
#         subset = Subset(trainset, client_data)
        
#         for (sample, lable) in subset:
#             group_lable_list.append(lable)

#         # print(lable_list)
#     sum = 0
#     for i in range(10):
#         count = group_lable_list.count(i)
#         print(count, end=" ")
#         sum += count
#     print(sum)
# s = subsets_to_target_set(subsets[0:3])

# print(s)


