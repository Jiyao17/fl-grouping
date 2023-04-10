
import matplotlib.pyplot as plt

x_axis = range(100, 1100, 100)

kldg = [2.23, 14.74, 59.80, 137.65, ]
kldg = [2.23, 14.74, 59.80,]

rg = [0, 0, 0, 0, 0, 0, 0.02, 0.027, 0.03, 0.0366]

cdg = [0, 0, 0, 0, 0, 0, 0, 0, 0.094, 0.12]

cvg = [0.06, 0.24, 0.54, 0.96, 1.52, 2.17, 2.96, 3.86, 4.85, 5.59]

# plot lines
plt.plot(x_axis, rg, label = "RG")
plt.plot(x_axis, cdg, label = "CDG")
plt.plot(x_axis[:len(kldg)], kldg, label = "KLDG")
plt.plot(x_axis, cvg, label = "CVG")

# set x, y labels
plt.xlabel('Number of Clients')
plt.ylabel('Time (s)')
plt.title('# of Clients vs. Time')

# set y limit
# plt.ylim(0, 10)
# set y to log scale
# plt.yscale('log')

# show legend
plt.legend()

plt.show()