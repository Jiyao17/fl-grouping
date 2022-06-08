
from cProfile import label
import numpy as np
import matplotlib.pyplot as plt

def p1():
    
    x = np.arange(0, 100, 1)
    y1 = np.array([0.3956,0.4062,0.4597,0.475,0.5045,0.5223,0.5504,0.5555,0.5616,0.5771,0.5832,0.5762,0.6068,0.6091,0.6,0.6061,0.6237,0.6215,0.6209,0.6397,0.6385,0.6334,0.6353,0.6537,0.6522,0.6561,0.6535,0.643,0.6484,0.6652,0.6585,0.6691,0.6706,0.6576,0.6864,0.6698,0.6863,0.6787,0.6976,0.6791,0.6857,0.6961,0.6766,0.7011,0.6842,0.6952,0.7084,0.6699,0.7114,0.7091,0.7239,0.7238,0.7264,0.7271,0.723,0.7257,0.7234,0.7246,0.7284,0.7233,0.7272,0.7259,0.7278,0.7256,0.7286,0.7247,0.7262,0.7268,0.7254,0.7273,0.7267,0.7288,0.7259,0.7271,0.7271,0.7295,0.7266,0.7266,0.7278,0.7273,0.7276,0.7285,0.7269,0.729,0.7275,0.7288,0.7263,0.7317,0.7286,0.7293,0.7276,0.726,0.7233,0.7296,0.7319,0.7297,0.7271,0.7316,0.7303,0.7279])
    plt.plot(x, y1, label='centralized', color='green')
    y2 = np.array([0.1599,0.2613,0.3608,0.4013,0.4233,0.4446,0.4592,0.4648,0.4754,0.4878,0.5024,0.5095,0.5134,0.5306,0.5387,0.545,0.5477,0.5544,0.561,0.5661,0.5669,0.5719,0.5804,0.5868,0.5848,0.5898,0.5922,0.5956,0.5938,0.6058,0.6056,0.6093,0.6098,0.6139,0.6121,0.6195,0.6244,0.6248,0.6302,0.6307,0.6294,0.6284,0.6299,0.6345,0.6355,0.6443,0.6406,0.6437,0.6438,0.6499,0.6485,0.65,0.6499,0.6493,0.6498,0.6524,0.6523,0.6531,0.6532,0.6519,0.6521,0.6512,0.6527,0.6527,0.6522,0.654,0.6523,0.6521,0.6531,0.6529,0.6557,0.654,0.6542,0.6542,0.6548,0.653,0.6546,0.6556,0.6559,0.6559,0.655,0.6551,0.6565,0.6567,0.6566,0.659,0.659,0.6581,0.6596,0.66,0.6588,0.6603,0.6594,0.6585,0.6581,0.6598,0.6598,0.6585,0.6598,0.6597])
    plt.plot(x, y2, label='iid', color='blue')
    y3 = np.array([0.1029,0.0998,0.1196,0.1956,0.2037,0.2515,0.2648,0.2844,0.2731,0.3117,0.2906,0.3148,0.3246,0.3131,0.3339,0.3301,0.3471,0.3514,0.3474,0.3679,0.3583,0.3683,0.3684,0.3684,0.3853,0.3724,0.3943,0.3707,0.3779,0.3865,0.3975,0.4025,0.4029,0.4079,0.4116,0.4092,0.4162,0.4187,0.4213,0.4321,0.4218,0.4418,0.4336,0.4417,0.4419,0.431,0.44,0.4537,0.4114,0.4535,0.4801,0.4805,0.4831,0.4814,0.4835,0.4832,0.4822,0.4857,0.4865,0.4885,0.4884,0.4868,0.4895,0.4886,0.4887,0.4902,0.4898,0.4901,0.4898,0.4921,0.4906,0.4908,0.4946,0.4947,0.4933,0.494,0.4947,0.494,0.4943,0.4925,0.4928,0.4974,0.4941,0.4936,0.4933,0.4958,0.4928,0.4956,0.4961,0.4958,0.4966,0.4949,0.4963,0.4971,0.4952,0.4976,0.4945,0.4949,0.4938,0.4968])
    plt.plot(x, y3, label='noniid', color='red')

    plt.rc('font', size=16)
    plt.subplots_adjust(0.15, 0.15, 0.95, 0.95)

    plt.xlabel('Training Round', fontsize=20)
    plt.ylabel('Accuracy', fontsize=20)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.grid(True)
    plt.legend()

    plt.savefig('no_selection.pdf')
    plt.savefig('no_selection.png')


def p2():
    full = np.array([0.101,0.2627,0.3495,0.3939,0.4185,0.4365,0.4519,0.4622,0.4785,0.4901,0.5014,0.504,0.5122,0.5256,0.5288,0.5298,0.5399,0.5421,0.5483,0.5577,0.5584,0.5665,0.5689,0.5698,0.5818,0.5872,0.5797,0.5962,0.5948,0.6012,0.6027,0.6062,0.6013,0.6025,0.6162,0.6132,0.6151,0.6192,0.62,0.6215,0.6203,0.6301,0.6326,0.6304,0.6314,0.6382,0.6411,0.6405,0.6431,0.6351,0.649,0.6506,0.6506,0.6524,0.6531,0.653,0.6534,0.6556,0.6542,0.6541,0.6551,0.658,0.6564,0.6546,0.6546,0.6565,0.6565,0.6577,0.658,0.6584,0.6579,0.6596,0.6574,0.6585,0.6579,0.6587,0.6605,0.6622,0.6616,0.66,0.662,0.662,0.6617,0.6642,0.6636,0.6656,0.6649,0.6644,0.6662,0.665,0.6662,0.6664,0.6653,0.6645,0.6653,0.6655,0.6658,0.6676,0.6673,0.6677])
    half = np.array([0.1007,0.2079,0.2585,0.3023,0.3414,0.359,0.3841,0.3964,0.4142,0.4301,0.4393,0.4481,0.4642,0.4738,0.4837,0.4898,0.4961,0.5064,0.5127,0.5163,0.5269,0.5313,0.5332,0.5341,0.5475,0.543,0.544,0.5444,0.5642,0.5548,0.5674,0.5696,0.5752,0.5774,0.5805,0.5767,0.5873,0.5855,0.5908,0.5957,0.5885,0.5918,0.6008,0.603,0.5978,0.6025,0.6066,0.6113,0.6101,0.6171,0.6205,0.6209,0.6224,0.6223,0.6234,0.6236,0.6261,0.6261,0.6246,0.6263,0.6257,0.6267,0.6273,0.6245,0.6271,0.6263,0.6289,0.6283,0.6278,0.63,0.6289,0.6297,0.6294,0.629,0.6297,0.6301,0.6295,0.6324,0.6288,0.6344,0.6323,0.6321,0.6329,0.6336,0.6326,0.6329,0.6331,0.634,0.6334,0.6344,0.6346,0.6342,0.6347,0.6367,0.6368,0.6358,0.6368,0.6361,0.6367,0.6358])
    single = np.array([0.1225,0.2831,0.3382,0.3667,0.387,0.3987,0.4138,0.4179,0.4264,0.4317,0.4322,0.438,0.4467,0.4477,0.4531,0.4511,0.457,0.4634,0.4703,0.4674,0.473,0.4805,0.4777,0.4838,0.4834,0.481,0.4804,0.4875,0.4823,0.4898,0.4903,0.4927,0.4928,0.4912,0.4937,0.4946,0.4952,0.4977,0.487,0.4964,0.4967,0.4933,0.4962,0.4963,0.4958,0.5017,0.5043,0.4995,0.5018,0.501,0.5068,0.5057,0.5042,0.5051,0.504,0.5024,0.5037,0.5041,0.5023,0.5016,0.5013,0.502,0.5,0.5012,0.5008,0.4987,0.5013,0.4981,0.5001,0.5001,0.5003,0.4988,0.4981,0.4992,0.4999,0.4986,0.4979,0.4985,0.4972,0.4969,0.4994,0.499,0.4992,0.5001,0.4969,0.4971,0.4975,0.4983,0.498,0.4965,0.4958,0.4975,0.4952,0.4968,0.4983,0.4945,0.4934,0.4954,0.4947,0.4958])

    x = np.arange(0, 100, 1)
    plt.plot(x, full, label='full', color='green')
    y2 = np.array([0.1599,0.2613,0.3608,0.4013,0.4233,0.4446,0.4592,0.4648,0.4754,0.4878,0.5024,0.5095,0.5134,0.5306,0.5387,0.545,0.5477,0.5544,0.561,0.5661,0.5669,0.5719,0.5804,0.5868,0.5848,0.5898,0.5922,0.5956,0.5938,0.6058,0.6056,0.6093,0.6098,0.6139,0.6121,0.6195,0.6244,0.6248,0.6302,0.6307,0.6294,0.6284,0.6299,0.6345,0.6355,0.6443,0.6406,0.6437,0.6438,0.6499,0.6485,0.65,0.6499,0.6493,0.6498,0.6524,0.6523,0.6531,0.6532,0.6519,0.6521,0.6512,0.6527,0.6527,0.6522,0.654,0.6523,0.6521,0.6531,0.6529,0.6557,0.654,0.6542,0.6542,0.6548,0.653,0.6546,0.6556,0.6559,0.6559,0.655,0.6551,0.6565,0.6567,0.6566,0.659,0.659,0.6581,0.6596,0.66,0.6588,0.6603,0.6594,0.6585,0.6581,0.6598,0.6598,0.6585,0.6598,0.6597])
    plt.plot(x, half, label='half', color='blue')
    y3 = np.array([0.1029,0.0998,0.1196,0.1956,0.2037,0.2515,0.2648,0.2844,0.2731,0.3117,0.2906,0.3148,0.3246,0.3131,0.3339,0.3301,0.3471,0.3514,0.3474,0.3679,0.3583,0.3683,0.3684,0.3684,0.3853,0.3724,0.3943,0.3707,0.3779,0.3865,0.3975,0.4025,0.4029,0.4079,0.4116,0.4092,0.4162,0.4187,0.4213,0.4321,0.4218,0.4418,0.4336,0.4417,0.4419,0.431,0.44,0.4537,0.4114,0.4535,0.4801,0.4805,0.4831,0.4814,0.4835,0.4832,0.4822,0.4857,0.4865,0.4885,0.4884,0.4868,0.4895,0.4886,0.4887,0.4902,0.4898,0.4901,0.4898,0.4921,0.4906,0.4908,0.4946,0.4947,0.4933,0.494,0.4947,0.494,0.4943,0.4925,0.4928,0.4974,0.4941,0.4936,0.4933,0.4958,0.4928,0.4956,0.4961,0.4958,0.4966,0.4949,0.4963,0.4971,0.4952,0.4976,0.4945,0.4949,0.4938,0.4968])
    plt.plot(x, single, label='single', color='red')

    plt.rc('font', size=16)
    plt.subplots_adjust(0.15, 0.15, 0.95, 0.95)

    plt.xlabel('Training Round', fontsize=20)
    plt.ylabel('Accuracy', fontsize=20)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.grid(True)
    plt.legend()

    plt.savefig('selection.pdf')
    plt.savefig('selection.png')

def p3():
    fedavg = np.array([7.80584,0.21873,0.03724,0.01371,0.01204,0.00839,0.00575,0.00486,0.00370,0.00285,0.00245,0.00241,0.00235,0.00230,0.00225,0.00221,0.00220,0.00219,0.00218,0.00217,0.00216,0.00215,0.00214,0.00213,0.00212,0.00212,0.00210,0.00210,0.00209,0.00209,0.00208,0.00207,0.00207,0.00206,0.00205,0.00205,0.00204,0.00203,0.00203,0.00202,0.00202,0.00201,0.00201,0.00200,0.00200,0.00199,0.00199,0.00198,0.00198,0.00197,0.00197,0.00197,0.00196,0.00196,0.00197,0.00197,0.00197,0.00197,0.00197,0.00197,0.00196,0.00196,0.00196,0.00196,0.00196,0.00196,0.00196,0.00196,0.00196,0.00196,0.00196,0.00196,0.00196,0.00196,0.00196,0.00196,0.00196,0.00196,0.00196,0.00196,0.00196,0.00196,0.00196,0.00196,0.00196,0.00195,0.00195,0.00195,0.00195,0.00195,0.00195,0.00195,0.00195,0.00195,0.00195,0.00195,0.00195,0.00195,0.00195,0.00195])
    grouping = np.array([9.16833,0.11449,0.00906,0.00642,0.00358,0.00304,0.00262,0.00234,0.00226,0.00225,0.00220,0.00216,0.00216,0.00215,0.00214,0.00213,0.00212,0.00211,0.00210,0.00209,0.00208,0.00208,0.00207,0.00207,0.00206,0.00206,0.00205,0.00205,0.00204,0.00204,0.00203,0.00203,0.00202,0.00202,0.00202,0.00201,0.00201,0.00201,0.00200,0.00200,0.00200,0.00200,0.00200,0.00199,0.00199,0.00198,0.00198,0.00198,0.00197,0.00197,0.00197,0.00197,0.00196,0.00196,0.00196,0.00196,0.00196,0.00196,0.00196,0.00196,0.00196,0.00196,0.00196,0.00196,0.00196,0.00196,0.00196,0.00196,0.00196,0.00196,0.00196,0.00196,0.00196,0.00196,0.00196,0.00196,0.00196,0.00196,0.00196,0.00196,0.00196,0.00196,0.00196,0.00196,0.00196,0.00196,0.00196,0.00195,0.00195,0.00195,0.00195,0.00195,0.00195,0.00195,0.00195,0.00195,0.00195,0.00195,0.00195,0.00195])

    x = np.arange(0, 100, 1)
    plt.ylim(0, 0.01)
    plt.xlim(0, 50)
    plt.plot(x, fedavg, label='FedAvg', color='red')
    plt.plot(x, grouping, label='Grouped', color='blue')

    plt.rc('font', size=16)
    plt.subplots_adjust(0.18, 0.15, 0.95, 0.95)

    plt.xlabel('Training Round', fontsize=20)
    plt.ylabel('Loss', fontsize=20)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.grid(True)
    plt.legend()

    plt.savefig('report.pdf')
    plt.savefig('report.png')

def nsf_grouping_comp():
    cvg_cvs = "0.1825 0.2216 0.2513 0.263 0.2784 0.2969 0.2997 0.3198 0.3238 0.3343 0.3357 0.3445 0.3466 0.358 0.3586 0.3595 0.3664 0.3724 0.378 0.3815 0.3889 0.3905 0.395 0.4014 0.4002 0.4032 0.4099 0.4094 0.4138 0.4188 0.4181 0.422 0.4255 0.4252 0.4259 0.4288 0.4293 0.4399 0.4354 0.4281 0.4368 0.439 0.43 0.4355 0.4372 0.4368 0.4442 0.4397 0.445 0.4466 0.4467 0.4481 0.4453 0.4447 0.4534 0.4514 0.4479 0.4549 0.4517 0.458"
    rg_cvs = "0.1759 0.2021 0.234 0.2528 0.2685 0.2813 0.2867 0.3013 0.3067 0.3221 0.3252 0.3173 0.3343 0.3408 0.3483 0.353 0.3407 0.3613 0.3537 0.3668 0.3576 0.3706 0.3782 0.3797 0.3699 0.3779 0.3957 0.3798 0.3898 0.3784 0.3933 0.386 0.3772 0.403 0.383 0.403 0.4103 0.4041 0.3961 0.4136 0.3978 0.3987 0.406 0.4113 0.4084 0.3977 0.4181 0.4207 0.422 0.4106 0.4096 0.4196 0.4291 0.428 0.4129 0.4309 0.415 0.4192 0.4301 0.4096"
    cvg_cvs_accu = np.array([float(x) for x in cvg_cvs.split(' ')])
    rg_cvs_accu = np.array([float(x) for x in rg_cvs.split(' ')])

    x = np.arange(0, 60, 1)
    plt.ylim(0, 0.5)
    plt.xlim(0, 50)
    plt.plot(x, cvg_cvs_accu, label='CVG-CVS', color='blue')
    plt.plot(x, rg_cvs_accu, label='RG-CVS', color='red')

    plt.rc('font', size=16)
    plt.subplots_adjust(0.18, 0.15, 0.95, 0.95)

    plt.xlabel('Training Round', fontsize=20)
    plt.ylabel('Accuracy', fontsize=20)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.grid(True)
    plt.legend()

    plt.savefig('grouping_comp.pdf')
    plt.savefig('grouping_comp.png')

    plt.clf()

def nsf_selection_comp():
    cvg_cvs = "0.1825 0.2216 0.2513 0.263 0.2784 0.2969 0.2997 0.3198 0.3238 0.3343 0.3357 0.3445 0.3466 0.358 0.3586 0.3595 0.3664 0.3724 0.378 0.3815 0.3889 0.3905 0.395 0.4014 0.4002 0.4032 0.4099 0.4094 0.4138 0.4188 0.4181 0.422 0.4255 0.4252 0.4259 0.4288 0.4293 0.4399 0.4354 0.4281 0.4368 0.439 0.43 0.4355 0.4372 0.4368 0.4442 0.4397 0.445 0.4466 0.4467 0.4481 0.4453 0.4447 0.4534 0.4514 0.4479 0.4549 0.4517 0.458"
    cvg_rs = "0.1687 0.2048 0.2244 0.2327 0.2477 0.2623 0.2737 0.2856 0.297 0.2994 0.3137 0.3173 0.3287 0.3261 0.3389 0.3358 0.3409 0.3525 0.3533 0.3549 0.3599 0.36 0.3671 0.3657 0.3676 0.3677 0.3714 0.3855 0.3859 0.3846 0.3856 0.3891 0.392 0.395 0.3894 0.3899 0.3986 0.4026 0.4007 0.4049 0.4066 0.4037 0.4089 0.408 0.4088 0.4141 0.4107 0.4146 0.4185 0.4235 0.4199 0.4241 0.4208 0.4205 0.4223 0.422 0.4297 0.4267 0.425 0.4282"
    cvg_cvs_accu = np.array([float(x) for x in cvg_cvs.split(' ')])
    cvg_rs_accu = np.array([float(x) for x in cvg_rs.split(' ')])

    x = np.arange(0, 60, 1)
    plt.ylim(0, 0.5)
    plt.xlim(0, 50)
    plt.plot(x, cvg_cvs_accu, label='CVG-CVS', color='blue')
    plt.plot(x, cvg_rs_accu, label='CVG-RS', color='red')

    plt.rc('font', size=16)
    plt.subplots_adjust(0.18, 0.15, 0.95, 0.95)

    plt.xlabel('Training Round', fontsize=20)
    plt.ylabel('Accuracy', fontsize=20)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.grid(True)
    plt.legend()

    plt.savefig('selection_comp.pdf')
    plt.savefig('selection_comp.png')

    plt.clf()

def nsf_gs_comp():
    gs5 = "0.1472 0.1788 0.2062 0.2115 0.236 0.2511 0.2546 0.2763 0.2682 0.2837 0.2812 0.2806 0.292 0.3069 0.3068 0.3171 0.3142 0.3201 0.3296 0.3145 0.3343 0.3389 0.3429 0.3609 0.3473 0.3498 0.3578 0.3278 0.35 0.3687 0.3647 0.3613 0.3469 0.3712 0.3748 0.3716 0.3715 0.3526 0.3789 0.3738 0.3708 0.3901 0.3855 0.3704 0.3834 0.3848 0.3846 0.3727 0.3877 0.3274 0.3969 0.3964 0.3942 0.3848 0.3816 0.3961 0.3924 0.4036 0.404 0.4087"
    gs10 = "0.2068 0.2409 0.2521 0.2644 0.2837 0.2881 0.2878 0.3128 0.3178 0.304 0.3192 0.3202 0.3373 0.3315 0.3514 0.3529 0.3439 0.3597 0.3667 0.365 0.367 0.3767 0.3792 0.3796 0.3923 0.3764 0.3882 0.3952 0.3919 0.3798 0.393 0.3826 0.3975 0.4051 0.3958 0.4005 0.3974 0.3957 0.4105 0.4054 0.4165 0.4233 0.4074 0.4068 0.4128 0.4176 0.4081 0.422 0.4123 0.4326 0.4122 0.4219 0.423 0.4128 0.4066 0.4303 0.4277 0.4298 0.4115 0.426"
    gs15 = "0.1934 0.2252 0.2474 0.2661 0.2744 0.2846 0.2886 0.31 0.3037 0.3226 0.3244 0.3303 0.3355 0.3386 0.3517 0.3542 0.3511 0.3548 0.3638 0.3607 0.3561 0.3671 0.3526 0.3722 0.3811 0.3832 0.3859 0.3895 0.3888 0.3772 0.3977 0.3966 0.394 0.3984 0.3854 0.4036 0.4047 0.4166 0.401 0.4035 0.4137 0.4061 0.424 0.4136 0.4125 0.4258 0.4351 0.4195 0.4133 0.4302 0.42 0.4296 0.4308 0.4381 0.4311 0.4312 0.4343 0.4188 0.4321 0.4302"
    gs20 = "0.1679 0.2071 0.2222 0.2544 0.2674 0.2727 0.2882 0.2981 0.2958 0.3008 0.297 0.3099 0.308 0.3145 0.3193 0.3015 0.3136 0.3289 0.3361 0.3392 0.3293 0.3349 0.3428 0.3526 0.3563 0.3625 0.3612 0.3615 0.3662 0.3741 0.3702 0.3725 0.3727 0.3753 0.3818 0.3779 0.3895 0.3909 0.3823 0.3675 0.3834 0.3987 0.3891 0.4008 0.3785 0.406 0.393 0.3911 0.3953 0.4164 0.3979 0.4177 0.389 0.4055 0.4112 0.4058 0.4128 0.4064 0.4266 0.4103"


    gs5_accu = np.array([float(x) for x in gs5.split(' ')])
    gs10_accu = np.array([float(x) for x in gs10.split(' ')])
    gs15_accu = np.array([float(x) for x in gs15.split(' ')])
    gs20_accu = np.array([float(x) for x in gs20.split(' ')])

    x = np.arange(0, 60, 1)
    plt.ylim(0, 0.5)
    plt.xlim(0, 50)
    plt.plot(x, gs5_accu, label='GS-5', color='red')
    plt.plot(x, gs10_accu, label='GS-10', color='green')
    plt.plot(x, gs15_accu, label='GS-15', color='blue')
    plt.plot(x, gs20_accu, label='GS-20', color='orange')

    plt.rc('font', size=16)
    plt.subplots_adjust(0.18, 0.15, 0.95, 0.95)

    plt.xlabel('Training Round', fontsize=20)
    plt.ylabel('Accuracy', fontsize=20)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)

    plt.grid(True)
    plt.legend()

    plt.savefig('gs_comp.pdf')
    plt.savefig('gs_comp.png')

    plt.clf()

def nsf_p_comp():
    p_rcv = "0.1825 0.2216 0.2513 0.263 0.2784 0.2969 0.2997 0.3198 0.3238 0.3343 0.3357 0.3445 0.3466 0.358 0.3586 0.3595 0.3664 0.3724 0.378 0.3815 0.3889 0.3905 0.395 0.4014 0.4002 0.4032 0.4099 0.4094 0.4138 0.4188 0.4181 0.422 0.4255 0.4252 0.4259 0.4288 0.4293 0.4399 0.4354 0.4281 0.4368 0.439 0.43 0.4355 0.4372 0.4368 0.4442 0.4397 0.445 0.4466 0.4467 0.4481 0.4453 0.4447 0.4534 0.4514 0.4479 0.4549 0.4517 0.458"
    p_ercv = "0.1888 0.2212 0.2509 0.2737 0.2916 0.3031 0.309 0.3251 0.3361 0.3472 0.3556 0.3549 0.3662 0.3652 0.3708 0.3792 0.377 0.3834 0.3846 0.3893 0.3899 0.3923 0.395 0.3951 0.3997 0.3999 0.4005 0.4091 0.4051 0.4114 0.4117 0.4139 0.414 0.4168 0.4151 0.421 0.4222 0.4207 0.423 0.4234 0.425 0.422 0.4231 0.4287 0.4283 0.4244 0.4273 0.4315 0.4344 0.4331 0.4321 0.4363 0.4357 0.4409 0.4356 0.4357 0.4393 0.4399 0.4393 0.439"
    p_esrcv = "0.2014 0.25 0.2652 0.283 0.3031 0.3212 0.3246 0.3386 0.3421 0.3652 0.3655 0.3762 0.3791 0.3952 0.388 0.4055 0.411 0.412 0.4162 0.4232 0.4229 0.4381 0.4359 0.4453 0.4434 0.4563 0.4582 0.4538 0.4619 0.4641 0.4741 0.4762 0.4712 0.4757 0.4849 0.4853 0.4846 0.4784 0.4881 0.4932 0.4939 0.5086 0.5072 0.5007 0.5101 0.511 0.5094 0.5214 0.5291 0.5166 0.5227 0.5291 0.5325 0.5335 0.5251 0.5413 0.5318 0.535 0.5408 0.547"
    p_r = "0.1687 0.2048 0.2244 0.2327 0.2477 0.2623 0.2737 0.2856 0.297 0.2994 0.3137 0.3173 0.3287 0.3261 0.3389 0.3358 0.3409 0.3525 0.3533 0.3549 0.3599 0.36 0.3671 0.3657 0.3676 0.3677 0.3714 0.3855 0.3859 0.3846 0.3856 0.3891 0.392 0.395 0.3894 0.3899 0.3986 0.4026 0.4007 0.4049 0.4066 0.4037 0.4089 0.408 0.4088 0.4141 0.4107 0.4146 0.4185 0.4235 0.4199 0.4241 0.4208 0.4205 0.4223 0.422 0.4297 0.4267 0.425 0.4282"
    p_rcv_accu = np.array([float(x) for x in p_rcv.split(' ')])
    p_ercv_accu = np.array([float(x) for x in p_ercv.split(' ')])
    p_esrcv_accu = np.array([float(x) for x in p_esrcv.split(' ')])
    p_r_accu = np.array([float(x) for x in p_r.split(' ')])

    x = np.arange(0, 60, 1)
    plt.ylim(0, 0.6)
    plt.xlim(0, 50)
    plt.plot(x, p_rcv_accu, label='P-RCV', color='blue')
    plt.plot(x, p_ercv_accu, label='P-ERCV', color='orange')
    plt.plot(x, p_esrcv_accu, label='P-ESRCV', color='green')
    plt.plot(x, p_r_accu, label='P-RAND', color='red')

    plt.rc('font', size=16)
    plt.subplots_adjust(0.18, 0.15, 0.95, 0.95)

    plt.xlabel('Training Round', fontsize=20)
    plt.ylabel('Accuracy', fontsize=20)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)

    plt.grid(True)
    plt.legend()
    
    plt.savefig('p_comp.pdf')
    plt.savefig('p_comp.png')

    plt.clf()

def nsf_gs_cost_comp():
    gs10 = "0.2068 0.2409 0.2521 0.2644 0.2837 0.2881 0.2878 0.3128 0.3178 0.304 0.3192 0.3202 0.3373 0.3315 0.3514 0.3529 0.3439 0.3597 0.3667 0.365 0.367 0.3767 0.3792 0.3796 0.3923 0.3764 0.3882 0.3952 0.3919 0.3798 0.393 0.3826 0.3975 0.4051 0.3958 0.4005 0.3974 0.3957 0.4105 0.4054 0.4165 0.4233 0.4074 0.4068 0.4128 0.4176 0.4081 0.422 0.4123 0.4326 0.4122 0.4219 0.423 0.4128 0.4066 0.4303 0.4277 0.4298 0.4115 0.426"
    gs20 = "0.1679 0.2071 0.2222 0.2544 0.2674 0.2727 0.2882 0.2981 0.2958 0.3008 0.297 0.3099 0.308 0.3145 0.3193 0.3015 0.3136 0.3289 0.3361 0.3392 0.3293 0.3349 0.3428 0.3526 0.3563 0.3625 0.3612 0.3615 0.3662 0.3741 0.3702 0.3725 0.3727 0.3753 0.3818 0.3779 0.3895 0.3909 0.3823 0.3675 0.3834 0.3987 0.3891 0.4008 0.3785 0.406 0.393 0.3911 0.3953 0.4164 0.3979 0.4177 0.389 0.4055 0.4112 0.4058 0.4128 0.4064 0.4266 0.4103"
    gs50 = ""
    gs10_accu = np.array([float(x) for x in gs10.split(' ')])
    gs20_accu = np.array([float(x) for x in gs20.split(' ')])
    gs50_accu = np.array([float(x) for x in gs50.split(' ')])
    
    gs10 = "12599.7549444 25229.082033899995 37794.851080199995 50309.42059169999 62925.50642219999 75369.45588569998 88258.7530269 100809.515313 113655.11629949999 126377.57357729998 138954.81838139996 151446.87775259998 164112.83899199998 176605.33973849998 189487.57487489996 202092.62632289997 214805.3733441 227288.60520929997 239994.2902257 252827.0913285 265487.75606429996 278122.8210327 290824.5336714 303361.6133232 316319.76501120004 329040.45678780007 341657.4253689001 354127.85735040007 366643.3096125001 379371.5047692001 392325.6840795001 404922.7907721001 417571.53837480006 429942.66091380006 442493.4231999 455312.9830437 467967.02715 480830.7245238 493622.91909899993 506070.3995648999 518729.7401747999 531504.2797379999 544460.6659247999 557512.3891763998 569969.1385235998 582342.0265637999 595296.6472493998 608083.1039456999 620827.1886131999 633150.6426197997 645736.2735545996 658381.0487795996 671115.8645657997 683682.5163626998 696380.2566236999 709169.3615718 721784.5646517 734401.5332328 746831.3586866999 759322.0939319998"
    gs20 = "12867.340434300004 25649.495435700006 38649.24846 51448.61709810001 64233.861726600015 77024.84423400002 89836.13000520002 102761.73197910002 115818.86379240002 128709.15574230002 141797.1838266 154634.0693652 167527.00956689997 180385.52249519996 193240.06304579997 205815.21303149997 218804.37304859995 231523.41138209996 244637.0392338 257252.7957471 269872.0832628 282641.87975580007 295475.23429200007 308253.85829100007 320927.43496860005 334000.8976680001 346616.21280600014 359398.80918270006 372142.12315770006 384907.9472730001 397751.4534411001 410494.3260408001 423305.6118120001 436282.8546960001 449042.9409324001 461983.5496665001 475048.18485990004 487768.5473193 500563.9435797 513078.18377400003 526114.1295729 538983.2355084001 551621.5021620002 564497.6701023001 577003.0827906001 589980.7670499 602812.3560849002 615623.2004808001 628555.8644595001 641638.1546649 654594.2115345 667436.834952 680022.5779448999 692826.8017112999 705367.5244236 718228.2442284 730821.0492261001 743548.0323150002 756498.3513057001 769321.5542100001"
    gs50 = ""
    gs10_cost = np.array([float(x) for x in gs10.split(' ')])
    gs20_cost = np.array([float(x) for x in gs20.split(' ')])
    gs50_cost = np.array([float(x) for x in gs50.split(' ')])
    
    x = np.arange(0, 60, 1)
    plt.ylim(0, 800000)
    plt.xlim(0, 50)
    plt.plot(x, gs5_accu, label='GS-5', color='red')
    plt.plot(x, gs10_accu, label='GS-10', color='green')
    plt.plot(x, gs15_accu, label='GS-15', color='blue')
    plt.plot(x, gs20_accu, label='GS-20', color='orange')

    plt.rc('font', size=16)
    plt.subplots_adjust(0.18, 0.15, 0.95, 0.95)

    plt.xlabel('Training Round', fontsize=20)
    plt.ylabel('Accuracy', fontsize=20)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)

    plt.grid(True)
    plt.legend()

    plt.savefig('gs_cost_comp.pdf')
    plt.savefig('gs_cost_comp.png')

    plt.clf()



if __name__ == '__main__':
    # nsf_report()
    nsf_grouping_comp()
    nsf_selection_comp()
    nsf_p_comp()
    nsf_gs_comp()
    # nsf_gs_cost_comp()