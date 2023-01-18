
# import numpy as np

# np.random.seed(0)

# group_num = 100
# sample_num = 20
# probs = np.random.random(group_num)
# probs = probs / np.sum(probs)
# print("raw probs", probs[:10])

# freqs = (1 - np.power(1-probs, sample_num))
# print("cal probs", freqs[:10])
# indices = range(0, group_num)
# # est_prob = freqs / np.sum(freqs)
# # print("est probs @ round 0:", est_prob[:10])
# # [0.01185996 0.01491999 0.01235044 0.0119748  0.00947989 0.01350013 0.00979985 0.0184296  0.01898016 0.00862991]

# # freqs = np.zeros(group_num)
# # [0.01186  0.01492  0.01235  0.011975 0.00948  0.0135   0.0098   0.01843 0.01898  0.00863 ]
# occur = np.zeros(group_num)
# for i in range(0, 10000):
#     selections = np.random.choice(indices, p=probs, size=sample_num, replace=False)

#     for selection in selections:
#         occur[selection] += 1
#         # freqs[selection] += 1

#     # if i % 10 == 9:
#     #     est_prob = freqs / np.sum(freqs)
#     #     print(f"round {i}:", est_prob[:10])

# # est_prob = freqs / np.sum(freqs)
# # print(f"round {i}:", est_prob[:10])
# print("real probs", occur[:10] / 10000)
# print("real probs", occur[:10] / sample_num / 10000)

# print(net)

# params_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
# for name, param in net.named_parameters():
#     print(param.dtype)
#     break

# print(params_num)

# from utils.data import load_dataset, TaskName, DatasetPartitioner

# trainset, testset = load_dataset(TaskName.CIFAR)

# partitioner = DatasetPartitioner(trainset, 1000, (10, 50), (0.1, 0.1))
# label_type_num = partitioner.label_type_num
# distributions = partitioner.get_distributions()
# subsets_indices = partitioner.get_subsets()
# partitioner.draw(10)

# from utils.model import SpeechCommand
# import torch

# net = SpeechCommand()

# optimizer = torch.optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
# print(optimizer.param_groups[0]['params'].keys())
# print(net)

import numpy as np
import matplotlib.pyplot as plt

data_range = (20, 200)
samples = []
means = []
stds = []
vars = []
gammas = []
weights = []
for i in range(0, 1000):
    samples.append(np.random.randint(data_range[0], data_range[1]))
    mean = np.mean(samples)
    std = np.std(samples)
    var = np.var(samples)
    gamma = np.sum([np.power(sample/mean, 2) for sample in samples])
    weight = [sample/(mean*len(samples)) for sample in samples]
    weight_var = np.var(weight)
    means.append(mean)
    stds.append(std)
    vars.append(var)
    gammas.append(gamma)
    weights.append(weight_var)

# plt.plot(means, label="mean")
# plt.plot(stds, label="std")
# plt.plot(gammas, label="gamma")
# plt.plot(weights[50:], label="weight")
plt.plot(vars, label="var")
plt.legend()
plt.savefig("test.png")
