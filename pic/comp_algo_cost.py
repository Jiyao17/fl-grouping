
import matplotlib.pyplot as plt

x_axis = range(100, 1100, 100)

kldg = [2.23, 14.74, 59.80, 137.65, ]
kldg = [2.23, 14.74, 59.80,]

rg = [0, 0, 0, 0, 0, 0, 0.02, 0.027, 0.03, 0.0366]

cdg = [0, 0, 0, 0, 0, 0, 0, 0, 0.094, 0.12]

cvg = [0.06, 0.24, 0.54, 0.96, 1.52, 2.17, 2.96, 3.86, 4.85, 5.59]

# plot lines
plt.plot(x_axis, rg, label = "RG", marker='s', markersize=10)
plt.plot(x_axis, cdg, label = "CDG", marker='^', markersize=10)
plt.plot(x_axis[:len(kldg)], kldg, label = "KLDG", marker='*', markersize=10)
plt.plot(x_axis, cvg, label = "CoVG", marker='o', markersize=10)

# set x, y labels
plt.xlabel('Number of Clients', fontsize=24)
plt.ylabel('Time (s)', fontsize=24)
# plt.title('# of Clients vs. Time')

# set y limit
# plt.ylim(0, 10)
# set y to log scale
# plt.yscale('log')
plt.rc('font', size=24)
plt.subplots_adjust(0.18, 0.16, 0.95, 0.96)

# plt.xlabel('Global Round', fontsize=24)
# plt.ylabel('Accuracy', fontsize=24)
plt.subplots_adjust(0.16, 0.16, 0.96, 0.96)

# plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
plt.xticks(fontsize=24)
plt.yticks(fontsize=24)
plt.grid(True)

# show legend
plt.legend()

plt.savefig('comp_algo_cost.png',)
plt.savefig('comp_algo_cost.pdf',)