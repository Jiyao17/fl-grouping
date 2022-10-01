
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline, BSpline
import numpy as np

colors = [(0.12156862745098039, 0.4666666666666667, 0.7058823529411765), # blue
(1.0, 0.4980392156862745, 0.054901960784313725), # orange
(0.17254901960784313, 0.6274509803921569, 0.17254901960784313), # green
(0.8392156862745098, 0.15294117647058825, 0.1568627450980392), # red
(0.5803921568627451, 0.403921568627451, 0.7411764705882353), # purple
(0.5490196078431373, 0.33725490196078434, 0.29411764705882354), # brown
(0.8901960784313725, 0.4666666666666667, 0.7607843137254902), # pink
(0.4980392156862745, 0.4980392156862745, 0.4980392156862745), # gray
(0.7372549019607844, 0.7411764705882353, 0.13333333333333333), # yellow
(0.09019607843137255, 0.7450980392156863, 0.8117647058823529) # cyan
]

root_data_dir = "/home/tuo28237/projects/fl-grouping/exp_data/"
sub_dirs = ["grouping/rg_rs/"] * 1 + ["grouping/rg_rs/fedprox/"] * 1 + ["grouping/rg_rs/scaffold/"] * 1 \
    + ["grouping/cvg_cvs/"] * 1 + ["grouping/cvg_cvs/fedprox/"] * 1 + ["grouping/cvg_cvs/scaffold/"] * 1 \

# sub_dirs = ["grouping/cvg_cvs/"] * 3
# sub_dirs = ["grouping/rg_rs/"] * 4
# sub_dirs = ["grouping/rg_cvs/"] * 4

# sub_dirs = ["debug/", "debug/", "debug/"]
# sub_dirs = ["grouping/rg_rs/", "grouping/rg_rs/", "grouping/cvg_cvs/", "grouping/cvg_cvs/"]
# marks = ["_fedprox", "_scaffold", "_sgd"]
# marks = ["_cv05_33", "_cv05_51", "_cv05_52", "_cv10_32", "_cv10_51", "_gs5",]
# marks = ["_cv01_51", "_cv05_51", "_cv05_52", "_cv05_32", "_cv05_33", "_cv10_51", "_cv10_32",]
marks = ["_alpha0.1_gs5_10*5", "_alpha0.1_gs5_10*5", "_alpha0.1_gs5_10*5" ] \
    + ["_alpha0.1_cv0.5_10*5", "_alpha0.1_cv0.5_10*5", "_alpha0.1_cv0.5_10*5"] 
# marks = ["_alpha0.1_gs10_5*2", "_alpha0.1_gs10_5*2", "_alpha0.1_gs10_5*2",] \
#     + ["_alpha0.1_cv0.5_5*2", "_alpha0.1_cv0.5_5*2", "_alpha0.1_cv0.5_5*2",] 

# marks = ["_alpha0.5_cv0.1_5*2", "_alpha0.5_cv0.5_5*2", "_alpha0.5_cv1.0_5*2"]
# marks = ["_alpha1_cv0.1_5*2", "_alpha1_cv0.5_5*2", "_alpha1_cv1.0_5*2"]

# marks = ["_cv05_53", "_scaffold_53", "_fedprox_53", "_cv05_53", "_cv05_53_cvg_cvs_scaffold", ]
# marks = ["_alpha0.1_gs5_5*2", "_alpha0.1_gs10_5*2", "_alpha0.1_gs20_5*2", "_alpha0.1_gs40_5*2", ]
# marks = ["RGRS-51", "RGRS-53", "RGRS-55", "RGRS-510"]
# exp_labels = [ mark[1:].replace("_", " ") for mark in marks]
# labels = ["GS=5", "GS=10", "GS=15", "GS=20"]
# labels = ["RG-RS51", "RG-RS32", "CVG-CVS32", "CVG-CVS51", "FedAvg15"]
colors = [ "black", "red", "blue", "green" ]
# fig_labels = ["Max CoV=0.1", "Max CoV=0.5", "Max CoV=1.0"]
fig_labels = ["FedAvg", "FedProx", "SCAFFOLD", "CVG", "CVG+FedProx", "CVG+SCAFFOLD"]

lines = []
for sub_dir, mark, label in zip(sub_dirs, marks, fig_labels):
    cost_filename = root_data_dir + sub_dir + "cost" + mark
    accu_filename = root_data_dir + sub_dir + "accu" + mark

    cost = open(cost_filename, "r").readlines()[-1].strip().split()
    if cost[0] != "" and cost[0][0] == "c":
        continue
    cost = [float(x) for x in cost if x != ""]
    accu = open(accu_filename, "r").readlines()[-1].strip().split()
    accu = [float(x) for x in accu if x != ""]

    min_dif = 1e9
    index = 0
    max_accu = 0
    for i, c in enumerate(cost):
        if c <= 1e6:
            if accu[i] > max_accu:
                max_accu = accu[i]
        else:
            break
    print(label)
    print(max_accu)
    # plt.plot(cost, accu, label=label)
    plt.plot(range(len(cost)), accu, label=label)
# 


# max_cost = 2000000
# plt.xlim(0, 40)
# plt.ylim(0.15, 0.4)

    # 300 represents number of points to make between T.min and T.max
    # cost_new = np.linspace(0, max_cost, 10000) 

    # spl = make_interp_spline(costs[i], accus[i], k=2)  # type: BSpline
    # accu_smooth = spl(cost_new)

    # plt.plot(cost_new, accu_smooth, label="GS="+mark[3:], color=colors[i])
    # plt.show()
        
    # "0.1472 0.1788 0.2062 0.2115 0.236 0.2511 0.2546 0.2763 0.2682 0.2837 0.2812 0.2806 0.292 0.3069 0.3068 0.3171 0.3142 0.3201 0.3296 0.3145 0.3343 0.3389 0.3429 0.3609 0.3473 0.3498 0.3578 0.3278 0.35 0.3687 0.3647 0.3613 0.3469 0.3712 0.3748 0.3716 0.3715 0.3526 0.3789 0.3738 0.3708 0.3901 0.3855 0.3704 0.3834 0.3848 0.3846 0.3727 0.3877 0.3274 0.3969 0.3964 0.3942 0.3848 0.3816 0.3961 0.3924 0.4036 0.404 0.4087 ",
    # "0.2068 0.2409 0.2521 0.2644 0.2837 0.2881 0.2878 0.3128 0.3178 0.304 0.3192 0.3202 0.3373 0.3315 0.3514 0.3529 0.3439 0.3597 0.3667 0.365 0.367 0.3767 0.3792 0.3796 0.3923 0.3764 0.3882 0.3952 0.3919 0.3798 0.393 0.3826 0.3975 0.4051 0.3958 0.4005 0.3974 0.3957 0.4105 0.4054 0.4165 0.4233 0.4074 0.4068 0.4128 0.4176 0.4081 0.422 0.4123 0.4326 0.4122 0.4219 0.423 0.4128 0.4066 0.4303 0.4277 0.4298 0.4115 0.426 ",
    # "0.1934 0.2252 0.2474 0.2661 0.2744 0.2846 0.2886 0.31 0.3037 0.3226 0.3244 0.3303 0.3355 0.3386 0.3517 0.3542 0.3511 0.3548 0.3638 0.3607 0.3561 0.3671 0.3526 0.3722 0.3811 0.3832 0.3859 0.3895 0.3888 0.3772 0.3977 0.3966 0.394 0.3984 0.3854 0.4036 0.4047 0.4166 0.401 0.4035 0.4137 0.4061 0.424 0.4136 0.4125 0.4258 0.4351 0.4195 0.4133 0.4302 0.42 0.4296 0.4308 0.4381 0.4311 0.4312 0.4343 0.4188 0.4321 0.4302 ",
    # "0.1679 0.2071 0.2222 0.2544 0.2674 0.2727 0.2882 0.2981 0.2958 0.3008 0.297 0.3099 0.308 0.3145 0.3193 0.3015 0.3136 0.3289 0.3361 0.3392 0.3293 0.3349 0.3428 0.3526 0.3563 0.3625 0.3612 0.3615 0.3662 0.3741 0.3702 0.3725 0.3727 0.3753 0.3818 0.3779 0.3895 0.3909 0.3823 0.3675 0.3834 0.3987 0.3891 0.4008 0.3785 0.406 0.393 0.3911 0.3953 0.4164 0.3979 0.4177 0.389 0.4055 0.4112 0.4058 0.4128 0.4064 0.4266 0.4103 ",
# plt.rc('font', size=24)
# plt.subplots_adjust(0.16, 0.18, 0.96, 0.96)

plt.xlabel('Cost', fontsize=24)
plt.ylabel('Global Round', fontsize=24)
plt.subplots_adjust(0.20, 0.18, 0.96, 0.96)

plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
plt.xticks(fontsize=24)
plt.yticks(fontsize=24)
plt.grid(True)
plt.legend()

# plt.rc('font', size=24)
# # plt.subplots_adjust(0.16, 0.18, 0.96, 0.96)

# plt.xlabel('Data/Group Size', fontsize=24)
# plt.ylabel('Time', fontsize=24)
# plt.xticks(fontsize=24)
# plt.yticks(fontsize=24)
# plt.grid(True)
# plt.legend()

plt.savefig("comp_m.png")
plt.savefig("comp_m.pdf")
