
import matplotlib.pyplot as plt
import numpy as np

def get_group_overhead(gs: list):
    group_coefs = [ 0.01629675, -0.02373668,  0.55442565]
    overheads = np.zeros(len(gs))

    for i in range(len(gs)):
        overheads[i] = group_coefs[0] * (gs[i] ** 2) + group_coefs[1] * gs[i] + group_coefs[2]

    return overheads


covg_gs =  [5.00, 5.29, 5.50, 6.07, 9.00, 9.75, 10.0, 12.0, 15.0]
covg_cov = [0.61, 0.58, 0.54, 0.46, 0.36, 0.36, 0.31, 0.24, 0.21]
covg_cost = get_group_overhead(covg_gs)

kldg_gs =  [5.50, 6.60, 8.25, 11.0, 16.5]
kldg_cov = [0.64, 0.53, 0.48, 0.42, 0.37]
kldg_cost = get_group_overhead(kldg_gs)

cdg_gs =  [5.50, 8.25, 11.0, 16.5,]
cdg_cov = [0.96, 0.70, 0.59, 0.47,]
cdg_cost = get_group_overhead(cdg_gs)

rg_gs =  [5.00, 8.00, 10.0, 13.0, 15.0]
rg_cov = [1.04, 0.77, 0.71, 0.64, 0.58]
rg_cost = get_group_overhead(rg_gs)

fig_type = 2
if fig_type == 1:
    plt.plot(rg_gs, rg_cov, label = "RG")
    plt.plot(covg_gs, covg_cov, label = "CoVG", marker='o', markersize=10)
    plt.plot(kldg_gs, kldg_cov, label = "KLDG")
    plt.plot(cdg_gs, cdg_cov, label = "CDG")


    # set x, y labels
    plt.xlabel('Avg. GS', fontsize=24)
    plt.ylabel('Avg. CoV', fontsize=24)
    # plt.title('GS v.s. CoV')
else:
    # CoV v.s. Cost
    plt.plot(rg_cost, rg_cov, label = "RG", marker='s', markersize=10)
    plt.plot(cdg_cost, cdg_cov, label = "CDG", marker='^', markersize=10)
    plt.plot(kldg_cost, kldg_cov, label = "KLDG", marker='*', markersize=10)
    plt.plot(covg_cost, covg_cov, label = "CoVG", marker='o', markersize=10)

    # set x, y labels
    plt.xlabel('Avg. CoV', fontsize=24)
    plt.ylabel('Avg. Group Overhead', fontsize=24)
    # plt.title('CoV v.s. Cost')

    
    

plt.rc('font', size=22)
plt.subplots_adjust(0.16, 0.16, 0.95, 0.96)

# plt.xlabel('Global Round', fontsize=24)
# plt.ylabel('Accuracy', fontsize=24)
# plt.subplots_adjust(0.16, 0.16, 0.96, 0.96)

# plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.grid(True)

# show legend
plt.legend()

plt.savefig('comp_gs_cov.png')
plt.savefig('comp_gs_cov.pdf')
