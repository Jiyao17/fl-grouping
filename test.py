
from utils.data import dataset_split_r_random_with_iid_datasets, load_dataset, subset_distribution
from torch.utils.data.dataset import Subset

trainset, testset = load_dataset("CIFAR", "../data/", "both")

indexes_list = dataset_split_r_random_with_iid_datasets(trainset, 100, 500, 4, 0.5)
d = [ Subset(trainset, indexes) for indexes in indexes_list ]

distribution = [ subset_distribution(subset) for subset in d ]
for i, distribution in enumerate(distribution):
    print("subset", i, ":", distribution)

