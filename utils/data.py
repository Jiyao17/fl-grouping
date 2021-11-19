
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
    within a dataset, r types might repeat
    """
    # each dataset contains r types of data
    categorized_index_list = dataset_categorize(dataset)
    indices_list = [[] for i in range(subset_num)]

    category_num = len(categorized_index_list)
    one_category_num = int(subset_size / r)
    counter = -1
    for i in range(subset_num):
        
        # choose r types randomly
        all_categories: list[int] = [ n for n in range(category_num)]
        choices: list[int] = []
        for j in range(r):
            choices = random.choices(all_categories, k=r)
            
        for choice in choices:
            # no enougn data in this category
            while len(categorized_index_list[choice]) < subset_size:
                # get another choice
                if len(all_categories) == 0:
                    raise "cannot creat groups"
                choice = all_categories.pop(random.randint(0, len(all_categories) - 1))

            # add choosed data
            indices_list[i] += categorized_index_list[choice][:one_category_num]
            # delete choosed data from dataset
            categorized_index_list[choice] = categorized_index_list[choice][one_category_num:]

        random.shuffle(indices_list[i])

    subsets = [ Subset(dataset, indices) for indices in indices_list ]
    return subsets

def dataset_split_r_random_distinct(dataset: Dataset, subset_num: int, subset_size, r: int) -> 'list[Dataset]':
    """
    within a dataset, r types are distinct
    """
    # each dataset contains r types of data
    categorized_index_list = dataset_categorize(dataset)
    indices_list = [[] for i in range(subset_num)]

    category_num = len(categorized_index_list)
    one_category_num = int(subset_size / r)
    counter = -1
    for i in range(subset_num):
        
        # choose r types randomly
        remnant_categories: list[int] = [ n for n in range(category_num)]
        choices: list[int] = []
        for j in range(r):
            choices.append(remnant_categories.pop(random.randint(0, len(remnant_categories) - 1)))
            
        for choice in choices:
            # no enougn data in this category
            while len(categorized_index_list[choice]) < subset_size:
                # get another choice
                if len(remnant_categories) == 0:
                    raise "cannot creat groups"
                choice = remnant_categories.pop(random.randint(0, len(remnant_categories) - 1))

            # add choosed data
            indices_list[i] += categorized_index_list[choice][:one_category_num]
            # delete choosed data from dataset
            categorized_index_list[choice] = categorized_index_list[choice][one_category_num:]

        random.shuffle(indices_list[i])

    subsets = [ Subset(dataset, indices) for indices in indices_list ]
    return subsets

def subset_select(fullset: set, subsets: 'list[Subset]', out_degree: int, group_num: int) -> 'list[list[int]]':
    targets = [subset.dataset.targets for subset in subsets]
    for i in range(len(targets)):
        if type(targets[i]) is not list:
            targets[i] = targets[i].tolist()
    # label set for all subsets
    targets_sets = [set(target) for target in targets]

    fullset_len = len(fullset)
    if out_degree > fullset_len:
        raise "Invalid out degree"
    subset_len = len(subsets[0])

    # subsets_type = [ set(subset) for subset in subsets ]
    groups: 'list[list[int]]' = [ [] ] * group_num
    unions: 'list[set[int]]' = [ set() ] * group_num
    
    for i in range(group_num):
        current_set = subsets.pop()
        groups[i].append(current_set)
        unions[i].union(set(current_set))

def subset_to_target_set(subset: Subset):
    target_set = set()
    for _, label in subset:
        target_set.add(label)
    
    return target_set

def subsets_to_target_set(subsets: 'list[Subset]'):
    target_set = set()
    for subset in subsets:
        target_set.union(subset_to_target_set(subset))

    return target_set

def find_next(group: 'list[Subset]', subsets: 'list[Subset]'):
    targets = subsets_to_target_set(group)

    for subset in subsets:
        target_set = subset_to_target_set(subset)