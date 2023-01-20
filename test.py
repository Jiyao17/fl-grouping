
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

# import numpy as np
# import matplotlib.pyplot as plt

# data_range = (20, 200)
# samples = []
# means = []
# stds = []
# vars = []
# gammas = []
# weights = []
# for i in range(0, 1000):
#     samples.append(np.random.randint(data_range[0], data_range[1]))
#     mean = np.mean(samples)
#     std = np.std(samples)
#     var = np.var(samples)
#     gamma = np.sum([np.power(sample/mean, 2) for sample in samples])
#     weight = [sample/(mean*len(samples)) for sample in samples]
#     weight_var = np.var(weight)
#     means.append(mean)
#     stds.append(std)
#     vars.append(var)
#     gammas.append(gamma)
#     weights.append(weight_var)

# # plt.plot(means, label="mean")
# # plt.plot(stds, label="std")
# # plt.plot(gammas, label="gamma")
# # plt.plot(weights[50:], label="weight")
# plt.plot(vars, label="var")
# plt.legend()
# plt.savefig("test.png")

accus0 = "0.1741 0.2651 0.3217 0.2432 0.2694 0.3318 0.3296 0.3431 0.3473 0.3548 0.3998 0.3276 0.3782 0.3802 0.4234 0.4314 0.4361 0.4301 0.4574 0.362 0.4389 0.4377 0.4554 0.4044 0.435 0.4569 0.3694 0.4474 0.4827 0.4594 0.4687 0.4589 0.4706 0.4435 0.4471 0.4965 0.4738 0.463 0.4789 0.4306 0.4819 0.5125 0.4885 0.5144 0.5015 0.459 0.5156 0.4793 0.5406 0.4976 0.483 0.5332 0.4923 0.5178 0.5059 0.4965 0.5029 0.5067 0.4999 0.5374 0.5509 0.5062 0.5264 0.5179 0.465 0.5811 0.5345 0.5325 0.5586 0.5894 0.5376 0.5483 0.5107 0.5435 0.563 0.539 0.5554 0.544 0.5082 0.5475 0.5345 0.5538 0.5765 0.5536 0.5899"
accus0 = [float(accu) for accu in accus0.split(" ")]
accus1 = "0.2622 0.2816 0.3056 0.342 0.3656 0.3754 0.3731 0.3549 0.3953 0.3937 0.4204 0.4128 0.4257 0.4585 0.4456 0.4634 0.4616 0.4311 0.4286 0.4373 0.4716 0.4508 0.4439 0.4737 0.4556 0.4669 0.4748 0.4751 0.4833 0.4814 0.4949 0.4242 0.5156 0.5064 0.4629 0.486 0.5016 0.5015 0.4647 0.4883 0.4755 0.5165 0.5085 0.4954 0.5169 0.5283 0.5475 0.488 0.5314 0.5399 0.5371 0.5198 0.5389 0.5316 0.5398 0.5573 0.5192 0.5398 0.5027 0.5476 0.5079 0.5254 0.5264 0.5212 0.5497 0.5716 0.5717 0.5429 0.5652 0.5495 0.5756 0.5497 0.5512 0.5383 0.5375 0.5526 0.5363 0.5469 0.5511 0.6007 0.5677"
accus1 = [float(accu) for accu in accus1.split(" ")]

import matplotlib.pyplot as plt

plt.plot(accus0, label="accu0")
plt.plot(accus1, label="accu1")
plt.legend()
plt.savefig("test.png")
