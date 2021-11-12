

from utils.data import dataset_split_r
from utils.task import TaskCIFAR


trainset, testset = TaskCIFAR.load_dataset("./data/")

subsets = dataset_split_r(trainset, 100, 100, 3)

lable_list = []
for (sample, lable) in subsets[5]:
    lable_list.append(lable)

print(lable_list)

sum = 0
for i in range(10):
    count = lable_list.count(i)
    print(count)
    sum += count

print(sum)