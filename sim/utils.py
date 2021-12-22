
import torchvision
import torchvision.transforms as tvtf
from torch import nn
from torch.utils.data.dataset import Dataset, Subset

import random



class ResBlock(nn.Module):
    def __init__(self, in_chann, chann, stride):
        super(ResBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_chann, chann, kernel_size=3, padding=1, stride=stride)
        self.bn1   = nn.BatchNorm2d(chann)
        
        self.conv2 = nn.Conv2d(chann, chann, kernel_size=3, padding=1, stride=1)
        self.bn2   = nn.BatchNorm2d(chann)

    def forward(self, x):
        y = self.conv1(x)
        y = self.bn1(y)
        y = F.relu(y)
        
        y = self.conv2(y)
        y = self.bn2(y)
        
        if (x.shape == y.shape):
            z = x
        else:
            z = F.avg_pool2d(x, kernel_size=2, stride=2)            

            x_channel = x.size(1)
            y_channel = y.size(1)
            ch_res = (y_channel - x_channel)//2

            pad = (0, 0, 0, 0, ch_res, ch_res)
            z = F.pad(z, pad=pad, mode="constant", value=0)

        z = z + y
        z = F.relu(z)
        return z


class BaseNet(nn.Module):
    
    def __init__(self, Block, n):
        super(BaseNet, self).__init__()
        self.Block = Block
        self.conv0 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.bn0   = nn.BatchNorm2d(16)
        self.convs  = self._make_layers(n)
        self.avgpool = nn.AvgPool2d(kernel_size=8, stride=1)
        self.fc = nn.Linear(64, 10)

    def forward(self, x):
        x = self.conv0(x)
        x = self.bn0(x)
        
        x = F.relu(x)
        
        x = self.convs(x)
        
        x = self.avgpool(x)

        x = x.view(x.size(0),-1)
        x = self.fc(x)
        
        return x

    def _make_layers(self, n):
        layers = []
        in_chann = 16
        chann = 16
        stride = 1
        for i in range(3):
            for j in range(n):
                if ((i > 0) and (j == 0)):
                    in_chann = chann
                    chann = chann * 2
                    stride = 2

                layers += [self.Block(in_chann, chann, stride)]

                stride = 1
                in_chann = chann

        return nn.Sequential(*layers)


class CIFARResNet(BaseNet):
    def __init__(self, n=3):
        super().__init__(ResBlock, n)


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

