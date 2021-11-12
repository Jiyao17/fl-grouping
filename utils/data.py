
from torch.utils.data.dataset import Dataset, Subset

import random


def dataset_categorize(dataset: Dataset) -> 'list[list[int]]':
    """
    return value:
    list[i] = list[int] = all indices for category i
    """
    targets = dataset.targets
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


def dataset_split_r(dataset: Dataset, subset_num: int, subset_size, r: int) -> 'list[Dataset]':
    """
    """
    # each dataset contains r types of data
    categorized_index_list = dataset_categorize(dataset)
    indices_list = [[] for i in range(subset_num)]

    # fill the dominant type of data
    category_num = len(categorized_index_list)
    counter = -1
    for i in range(subset_num):
        for j in range(r):
            counter += 1
            counter %= category_num

            indices_list[i] += categorized_index_list[counter][:subset_size]
            categorized_index_list[counter] = categorized_index_list[counter][subset_size:]

    subsets = [ Subset(dataset, indices) for indices in indices_list ]
    return subsets
