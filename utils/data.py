
import torchvision
import torchvision.transforms as tvtf
from torch.utils.data.dataset import Dataset, Subset

import numpy as np

import random


def load_dataset(data_path: str, type: str="both"):
    # enhance
    # Use the torch.transforms, a package on PIL Image.
    transform_enhanc_func = tvtf.Compose([
        tvtf.RandomHorizontalFlip(p=0.5),
        tvtf.RandomCrop(32, padding=4, padding_mode='edge'),
        tvtf.ToTensor(),
        tvtf.Lambda(lambda x: x.mul(255)),
        tvtf.Normalize([125., 123., 114.], [1., 1., 1.])
        ])

    # transform
    transform_func = tvtf.Compose([
        tvtf.ToTensor(),
        tvtf.Lambda(lambda x: x.mul(255)),
        tvtf.Normalize([125., 123., 114.], [1., 1., 1.])
        ])

    trainset, testset = None, None
    if type != "test":
        trainset = torchvision.datasets.CIFAR10(root=data_path, train=True,
            download=True, transform=transform_enhanc_func)
    if type != "train":
        testset = torchvision.datasets.CIFAR10(root=data_path, train=False,
            download=True, transform=transform_func)

    return (trainset, testset)

def get_targets(dataset: Dataset) -> list:
    targets = dataset.targets
    if type(targets) is not list:
        targets = targets.tolist()
    
    return targets

def get_targets_set(dataset: Dataset) -> list:
    targets = dataset.targets
    if type(targets) is not list:
        targets = targets.tolist()
    targets_list = list(set(targets))
    # can be deleted, does not matter but more clear if kept
    targets_list.sort()

    return targets_list



def dataset_categorize(dataset: Dataset) -> 'list[list[int]]':
    """
    return value:
    list[i] = list[int] = all indices for category i
    """
    targets = dataset.targets
    if type(dataset.targets) is not list:
        targets = targets.tolist()
    targets_list = list(set(targets))
    # can be deleted, does not matter but more clear if kept
    targets_list.sort()

    indices_by_lable = [[] for target in targets_list]
    for i, target in enumerate(targets):
        category = targets_list.index(target)
        indices_by_lable[category].append(i)

    # randomize
    for indices in indices_by_lable:
        random.shuffle(indices)

    # subsets = [Subset(dataset, indices) for indices in indices_by_lable]
    return indices_by_lable

def dataset_split_r_random(dataset: Dataset, subset_num, subset_size, r: int) \
     -> 'list[list[int]]':
    """
    within a dataset, r types are distinct

    revised and tested
    """

    if subset_size * subset_num > len(dataset):
        raise "Cannot partition dataset as required."

    # each dataset contains r types of data
    categorized_index_list = dataset_categorize(dataset)
    shard_size = int(subset_size / r)

    shards_list = []
    for one_category in categorized_index_list:
        shard_counter = 0
        while (shard_counter + 1) * shard_size <= len(one_category):
            shards_list.append(one_category[shard_counter*shard_size: (shard_counter+1)*shard_size])
            shard_counter += 1
        # shards_list.append(one_category[shard_counter*shard_size:])


    random.shuffle(shards_list)

    indices_list = []
    for i in range(subset_num):
        indices = []
        for j in range(r):
            indices += shards_list[i*r + j]
        indices_list.append(indices)
    
    return indices_list


