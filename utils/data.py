
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

def dataset_split_r(dataset: Dataset, subset_num: int, subset_size, r: int) \
    -> 'list[list[int]]':
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

    # subsets = [ Subset(dataset, indices) for indices in indices_list ]
    return indices_list

def dataset_split_r_random(dataset: Dataset, subset_num: int, subset_size, r: int) \
    -> 'list[list[int]]':
    """
    within a dataset, r types might repeat
    under development
    """
    # each dataset contains r types of data
    categorized_index_list = dataset_categorize(dataset)
    indices_list = [[] for i in range(subset_num)]

    category_num = len(categorized_index_list)
    one_category_num = int(subset_size / r)
    counter = -1
    all_categories: list[int] = [ n for n in range(category_num)]
    for i in range(subset_num):
        
        # choose r types randomly
        choice = random.choice(all_categories)
            
        while len(indices_list[i]) < subset_size:
            # no enougn data in this category
            while len(categorized_index_list[choice]) < one_category_num:
                # get another choice
                if len(all_categories) == 0:
                    for indexes in categorized_index_list:
                        print(len(indexes))
                    raise "cannot creat groups"
                choice = random.randint(0, len(all_categories) - 1)

            # add choosed data
            indices_list[i] += categorized_index_list[choice][:one_category_num]
            # delete choosed data from dataset
            categorized_index_list[choice] = categorized_index_list[choice][one_category_num:]

        random.shuffle(indices_list[i])

    # subsets = [ Subset(dataset, indices) for indices in indices_list ]
    return indices_list

def dataset_split_r_random_distinct(dataset: Dataset, subset_size, r: int) \
     -> 'list[list[int]]':
    """
    within a dataset, r types are distinct
    """
    # each dataset contains r types of data
    categorized_index_list = dataset_categorize(dataset)
    indices_list = []

    category_num = len(categorized_index_list)
    one_category_num = int(subset_size / r)
    more_subset = True
    while more_subset:
        new_indices: list[int] = []
        # choose r types randomly
        remnant_categories: list[int] = [ i for i in range(category_num)]
        choices: list[int] = []
        for j in range(r):
            choices.append(remnant_categories.pop(random.randint(0, len(remnant_categories) - 1)))
            
        for choice in choices:
            # no enougn data in this category
            while len(categorized_index_list[choice]) < one_category_num:
                # get another choice
                if len(remnant_categories) == 0:
                    more_subset = False
                    break
                choice = remnant_categories.pop(random.randint(0, len(remnant_categories) - 1))
            if more_subset == False:
                break

            # add choosed data
            new_indices += categorized_index_list[choice][:one_category_num]
            # delete choosed data from dataset
            categorized_index_list[choice] = categorized_index_list[choice][one_category_num:]

        if more_subset == True:
            random.shuffle(new_indices)
            indices_list.append(new_indices)

    # subsets = [ Subset(dataset, indices) for indices in indices_list ]
    return indices_list

def grouping(indices_list: 'list[list[int]]', targets: 'list[int]', out_degree: int) \
    -> 'list[list[int]]':
    # sets = datasets_to_target_sets(subsets)
    sets: list[set] = []
    for indices in indices_list:
        new_set = set()
        for index in indices:
            new_set.add(targets[index])
        sets.append(new_set)

    groups: 'list[list[int]]' = []
    unions: 'list[set[int]]' = []
    # group_labels: set = set()
    
    more_group = True
    while more_group:
        new_group = []
        new_set = set()
        while len(new_set) < out_degree:
            pos = find_next(new_set, sets)
            if pos >= 0:
                new_group += indices_list[pos]
                new_set = new_set.union(sets[pos])
                indices_list.pop(pos)
                sets.pop(pos)
            else:
                more_group = False
                break
        if more_group:
            groups.append(new_group)
            unions.append(new_set)
        else:
            break
    # replace index with subset
    # for group in groups:
    #     for i in range(len(group)):
    #         group[i] = subsets[group[i]]

    return groups

def dataset_to_target_set(subset: Subset) -> set:
    target_set = set()
    for _, label in subset:
        target_set.add(label)
    
    return target_set

def datasets_to_target_sets(subsets: 'list[Subset]') -> 'list[set]':
    target_sets: 'list[set]' = []
    for subset in subsets:
        new_set = dataset_to_target_set(subset)
        target_sets.append(new_set)

    return target_sets

def find_next(cur_set: set, subunions: 'list[set]') -> bool:
    max_len = 0
    pos = -1
    for i, subunion in enumerate(subunions):
        cur_len = len(subunion.difference(cur_set))
        if cur_len > max_len:
            pos = i
            max_len = cur_len
    
    return pos

def regroup(groups: 'list[list[int]]', new_group_num: int):

    new_size = len(groups) // new_group_num
    real_groups: 'list[list[int]]' = []
    counter = 0
    for i in range(new_group_num):
        new_group_indices = []
        for j in range(new_size):
            new_group_indices += groups[counter]
            counter += 1
        real_groups.append(new_group_indices)

    for i in range(new_group_num*new_size, len(groups) - 1):
        rand = random.randint(0, len(real_groups) - 1)
        real_groups[rand] += groups[i]
        
    return real_groups

