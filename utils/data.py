
import torchvision
import torchvision.transforms as tvtf
from torch.utils.data.dataset import Dataset, Subset

import numpy as np

import random
import math

def load_dataset_CIFAR(data_path: str, dataset_type: str="both"):
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
    if dataset_type != "test":
        trainset = torchvision.datasets.CIFAR10(root=data_path, train=True,
            download=True, transform=transform_enhanc_func)
    if dataset_type != "train":
        testset = torchvision.datasets.CIFAR10(root=data_path, train=False,
            download=True, transform=transform_func)

    return (trainset, testset)

def load_dataset(dataset_name: str, data_path: str, dataset_type: str="both"):
    if dataset_name == 'CIFAR':
        return load_dataset_CIFAR(data_path, dataset_type)

def get_targets(dataset: Dataset) -> list:
    targets = dataset.targets
    if type(targets) is not list:
        targets = targets.tolist()
    
    return targets

def get_targets_set_as_list(dataset: Dataset) -> list:
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

def dataset_split_r_random_with_iid_datasets(
    dataset: Dataset, subset_num: int, subset_size: int,
    r: int, iid_proportion: float=0.5) \
     -> 'list[list[int]]':
    """
    similar to dataset_split_r_random
    but partial datasets are iid
    thus some datasets are noniid with degree r,
    some are iid (occupy iid_propotion of total datasets)
    """
    
    if subset_size * subset_num > len(dataset):
        raise "Cannot partition dataset as required."
    
    iid_num = int(subset_num * iid_proportion)
    noniid_num = subset_num - iid_num

    # each dataset contains r types of data
    categorized_index_list = dataset_categorize(dataset)
    shard_size = math.gcd(int(subset_size / len(categorized_index_list)), int(subset_size/r))
    shards_lists = [ [] for i in range(len(categorized_index_list)) ]
    for i, one_category in enumerate(categorized_index_list):
        shard_counter = 0
        while (shard_counter + 1) * shard_size <= len(one_category):
            shards_lists[i].append(one_category[shard_counter*shard_size: (shard_counter+1)*shard_size])
            shard_counter += 1
        # shards_list.append(one_category[shard_counter*shard_size:])

    # iid datasets
    iid_datasets = []
    for i in range(iid_num):
        iid_set = []
        shard_index = 0
        # select one shard of each kind of data
        while len(iid_set) < subset_size:
            # bug: if len(shards_lists[shard_index]) == 0, then shard_index += 1
            shard = shards_lists[shard_index].pop()
            iid_set += shard
            shard_index = (shard_index + 1) % len(shards_lists)
        iid_datasets.append(iid_set)
    
    # noniid datasets
    noniid_datasets = []
    for i in range(noniid_num):
        noniid_set = []
        total_shard_num = subset_size/shard_size
        one_type_shard_num = int(total_shard_num / r)

        counter = 0
        # select r knids of data
        for j in range(r):
            counter = 0
            target_type = random.randint(0, len(shards_lists)-1)
            # no more shard of this type
            while len(shards_lists[target_type]) < one_type_shard_num:
                counter += 1
                target_type = (target_type + 1) % len(shards_lists)
                if(counter > 10):
                    break
            # select shards of this type
            for k in range(one_type_shard_num):
                shard = shards_lists[target_type].pop()
                noniid_set += shard
        if counter > 10:
            break
        else:
            noniid_datasets.append(noniid_set)
    
    # combine iid and noniid datasets
    indices_list = iid_datasets + noniid_datasets
    return indices_list

def dataset_split_r_random(dataset: Dataset, subset_num, subset_size, r: int) \
     -> 'list[list[int]]':
    """
    within a dataset, r types may not be distinct

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

def subset_distribution(subset: Subset) -> np.ndarray:
        lable_list = get_targets_set_as_list(subset.dataset)
        labels = np.zeros(shape=(len(lable_list),), dtype=np.int32,)
        indices = subset.indices
        for index in indices:
            (image, label) = subset.dataset[index]
            labels[label] += 1

        return labels

def partition_distribution(d: 'list[Subset]') -> 'list[np.ndarray]':
    """
    return a numpy array of shape (len(d), len(d[0].targets))
    """
    lable_list = get_targets_set_as_list(d[0].dataset)
    labels_on_clients = [ np.zeros(shape=(len(lable_list),), dtype=np.int32,) for i in range(len(d)) ]
    for i, subset in enumerate(d):
        indices = subset.indices
        for index in indices:
            (image, label) = subset.dataset[index]
            labels_on_clients[i][label] += 1

    return labels_on_clients
