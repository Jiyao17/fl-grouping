
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

rg_gs =  [5.00, 10.0, 15.0]
rg_cov = [1.04, 0.71, 0.58]
rg_cost = get_group_overhead(rg_gs)

fig_type = 2
if fig_type == 1:
    plt.plot(covg_gs, covg_cov, label = "COVG")
    plt.plot(kldg_gs, kldg_cov, label = "KLDG")
    plt.plot(cdg_gs, cdg_cov, label = "CDG")
    plt.plot(rg_gs, rg_cov, label = "RG")

    # set x, y labels
    plt.xlabel('Avg. GS')
    plt.ylabel('Avg. CoV')
    plt.title('GS v.s. CoV')
else:
    # CoV v.s. Cost
    plt.plot(covg_cov, covg_cost, label = "COVG")
    plt.plot(kldg_cov, kldg_cost, label = "KLDG")
    plt.plot(cdg_cov, cdg_cost, label = "CDG")
    plt.plot(rg_cov, rg_cost, label = "RG")

    # set x, y labels
    plt.xlabel('Avg. CoV')
    plt.ylabel('Avg. Cost')
    plt.title('CoV v.s. Cost')

    
    


# show legend
plt.legend()

plt.show()
