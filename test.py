
from tempfile import _TemporaryFileWrapper
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

samples0 = np.array([10, 10, 0, 0, 0, 0, 0, 0, 0, 0])
samples1 = np.array([5, 5, 0, 0, 0, 0, 0, 0, 0, 0])
samples2 = np.array([20, 20, 0, 0, 0, 0, 0, 0, 0, 0])

samples = samples2
std = np.std(samples)
muta = std / np.average(samples)

print(std, muta)