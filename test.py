
import numpy as np

# from utils.data import dataset_split_r_random_with_iid_datasets, load_dataset, subset_distribution
# from torch.utils.data.dataset import Subset

# trainset, testset = load_dataset("CIFAR", "../data/", "both")

# indexes_list = dataset_split_r_random_with_iid_datasets(trainset, 100, 50, 2, 0.5)
# d = [ Subset(trainset, indexes) for indexes in indexes_list ]

# distribution = [ subset_distribution(subset) for subset in d ]
# for i, distribution in enumerate(distribution):
#     print("subset", i, ":", distribution)

import enum

# class Mode(enum.Enum):
#     iid = 1
#     noniid = 2
#     random = 3
    
# print(Mode.iid.name)
# print(Mode.iid.value)
# print(type(Mode))

# arr = [ i*2 for i in range(10) ]
# arr = np.array(arr)
# indices = [ 1, 3, 5, 5 ]

# print(np.mean(arr[indices]))

# Q = [ 13, 17, 7, 11 ]
# stds = [ 1, 0, 1, 0]
# seq = [ i for i in range(len(Q))]

# Q_sorted = np.array(sorted(zip(Q, stds, seq)))

# index = np.array([ i for i, q in enumerate(Q_sorted) if q[1] == 0 ])
# print(index)

# Q_sorted = Q_sorted[index]

# for (q, std, seq) in Q_sorted:
#     print(type(int(seq)))
#     print(q, std, seq)

# print(Q_sorted[index])
# print(Q_sorted[~index])

# samples0 = np.array([10, 10, 0, 0, 0, 0, 0, 0, 0, 0])
# samples1 = np.array([5, 5, 0, 0, 0, 0, 0, 0, 0, 0])
# samples2 = np.array([20, 20, 0, 0, 0, 0, 0, 0, 0, 0])

# samples = samples2
# std = np.std(samples)
# muta = std / np.average(samples)

# print(std, muta)

from utils.data import DatasetPartitioner, load_dataset_CIFAR

sample_num = 10
trainset, testset = load_dataset_CIFAR("./data/", "both")
dp = DatasetPartitioner(trainset, 1000, (10, 50), 0.1, 0)
dp.draw(20, "./pic/debug.png")
# sigmas = np.std(dp.distributions[:sample_num], axis=1)
# sampled = np.zeros((10,))
# for distribution in dp.distributions[:sample_num]:
#     sampled += distribution
print(dp.distributions[:sample_num])
print(np.sum(dp.distributions[:sample_num], axis=1)) 
print(np.sum(dp.distributions[:sample_num]))

print(np.sum(dp.distributions[:sample_num], axis=1) / np.sum(dp.distributions[:sample_num]))
# print(sigmas)
# print(np.std(sampled) / np.average(sampled))

# import numpy as np

# def sample(population, probs, k) -> 'list[int]':
#     """
#     select groups for the current iteration
#     list of group numbers (in self.groups)
#     """
    
#     selected_groups: 'list[int]' = np.random.choice(population, replace=False, size=k, p=probs)
#     return selected_groups

# population = [0, 1, 2, 3, 4]
# probs = [0.2, 0.05, 0.3, 0.1, 0.35]

# times = [0, 0, 0, 0, 0]
# for i in range(5):
#     selected = sample(population, probs, 3)
#     # print(selected)
#     for s in selected:
#         times[s] += 1

# print(times)

# total = sum(times)
# for i, t in enumerate(times):
#     times[i] = t / total

# print(times)


