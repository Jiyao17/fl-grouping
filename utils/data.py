
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

def cluster()

def grouping(d: 'list[Subset]', D, B) \
    -> 'np.ndarray':
    """
    return
    G: grouping matrix
    """

    G = np.ndarray(())

    # group clients by delay to servers
    groups: list[list] = [ [] for i in range(B.shape[0]) ]
    server_indices = np.argmin(D, 1)
    for i, server in enumerate(server_indices):
        groups[server].append(i)

    group_num = 0
