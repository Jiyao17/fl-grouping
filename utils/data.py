
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


def find_next(cur_set: set, subunions: 'list[set]') -> int:
    """
    return the index if can add more categories

    return -1 if cannot add more set
    """
    max_len = 0
    pos = -1
    for i, subunion in enumerate(subunions):
        cur_len = len(subunion.difference(cur_set))
        if cur_len > max_len:
            pos = i
            max_len = cur_len
    
    return pos

def clustering(d: 'list[Subset]', group: 'list[int]') -> 'list[list[int]]':
    # sets = datasets_to_target_sets(subsets)
    sets: list[set] = []
    targets = get_targets(d[0].dataset)
    for client in group:
        new_set = set()
        for index in d[client].indices:
            new_set.add(targets[index])
        sets.append(new_set)

    groups: 'list[list[int]]' = []
    # indicates unions of current groups
    unions: 'list[set[int]]' = []
    # group_labels: set = set()
    targets_set = set(targets)
    while len(sets) > 0:
        # try to get a new group
        new_group: 'list[list[int]]' = []
        new_set = set()

        pos = find_next(new_set, sets)
        while pos != -1:
            new_group.append(group[pos])
            new_set = new_set.union(sets[pos])
            group.pop(pos)
            sets.pop(pos)

            pos = find_next(new_set, sets)

        groups.append(new_group)
        unions.append(new_set)
    # replace index with subset
    # for group in groups:
    #     for i in range(len(group)):
    #         group[i] = subsets[group[i]]
    return groups


def grouping_default(d: 'list[Subset]', D, B) \
    -> 'tuple[np.ndarray, np.ndarray, np.ndarray]':
    """
    Assign each client to its nearest server
    Clustering clients connected to the same server
    return group and accesory information
    G: grouping matrix
    A: initial assignment matrix, delay and bandwidth not considered
    M: g*s, delay, groups to servers
    """

    # group clients by delay to servers
    groups: list[list[int]] = [ [] for i in range(B.shape[0]) ]
    # clients to their nearest servers
    server_indices = np.argmin(D, 1)
    for i, server in enumerate(server_indices):
        groups[server].append(i)

    group_num = 0
    clusters_list = []
    # cluster clients to the same server
    for group in groups:
        clusters = clustering(d, group)
        clusters_list.append(clusters)
        group_num += len(clusters)

    # get initial G A M without sufficing l & B
    G = np.zeros((len(d), group_num), int)
    A = np.zeros((group_num, B.shape[0]), int)
    # G_size = np.zeros((group_num,))
    M = -1 * np.ones((group_num, B.shape[0]))

    group_counter = 0
    for server, clusters in enumerate(clusters_list):
        for cluster in clusters:
            max_delay = -1
            for client in cluster:
                G[client][group_counter] = 1

                # group delay
                delay = D[client][server]
                if delay > max_delay:
                    max_delay = delay

            A[group_counter][server] = 1
            # G_size[group_counter] = len(cluster)
            M[group_counter][server] = max_delay
            group_counter += 1

    return G, A, M
