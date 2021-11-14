
from torch.utils.data.dataset import Dataset, Subset

import random


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


def dataset_split_r(dataset: Dataset, subset_num: int, subset_size, r: int) -> 'list[Dataset]':
    """
    """
    # each dataset contains r types of data
    categorized_index_list = dataset_categorize(dataset)
    indices_list = [[] for i in range(subset_num)]

    # fill the dominant type of data
    category_num = len(categorized_index_list)
    one_category_num = int(subset_size / r)
    counter = -1
    for i in range(subset_num):
        for j in range(r):
            counter += 1
            counter %= category_num

            indices_list[i] += categorized_index_list[counter][:one_category_num]
            categorized_index_list[counter] = categorized_index_list[counter][one_category_num:]

        random.shuffle(indices_list[i])

    subsets = [ Subset(dataset, indices) for indices in indices_list ]
    return subsets


def dataset_split_r_random(dataset: Dataset, subset_num: int, subset_size, r: int) -> 'list[Dataset]':
    """
    """
    # each dataset contains r types of data
    categorized_index_list = dataset_categorize(dataset)
    indices_list = [[] for i in range(subset_num)]

    # fill the dominant type of data
    category_num = len(categorized_index_list)
    one_category_num = int(subset_size / r)
    counter = -1
    for i in range(subset_num):
        
        for j in range(r):
            # choose r types randomly
            remnant_categories: list[int] = [ n for n in range(category_num)]
            choices: list[int] = random.choices(remnant_categories, k=r)
                
            for choice in choices:
                # no enougn data in this category
                while len(categorized_index_list[choice]) < subset_size:
                    # get another choice
                    choice = random.randint(0, category_num - 1)
                # add choosed data
                indices_list[i] += categorized_index_list[choice][:one_category_num]
                # delete choosed data from dataset
                categorized_index_list[choice] = categorized_index_list[counter][one_category_num:]

        random.shuffle(indices_list[i])

    subsets = [ Subset(dataset, indices) for indices in indices_list ]
    return subsets
