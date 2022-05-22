
import matplotlib.pyplot as plt


def read_last_line_data(path: str) -> 'list[list[float]]':
    with open(path, "r") as f:
        lines = f.readlines()
        data = lines[-1].strip().split(" ")
        # print(data)
        data = [float(x) for x in data]
        return data

def draw_accu_by_round(accus, labels):
    for i, accu in enumerate(accus):
        xaxis = range(len(accu))
        plt.plot(xaxis, accu, label=labels[i])
    plt.legend()
    plt.savefig("./pics/accu_by_round.png")
    plt.clf()

def draw_loss_by_round(losses, labels):
    for i, loss in enumerate(losses):
        xaxis = range(len(loss))
        plt.plot(xaxis, loss, label=labels[i])
    plt.legend()
    plt.savefig("./pics/loss_by_round.png")
    plt.clf()

def draw_accu_by_cost(accus, costs, labels):
    for i, (accu, cost) in enumerate(zip(accus, costs)):
        plt.plot(cost, accu, label=labels[i])
    plt.legend()
    plt.savefig("./pics/accu_by_cost.png")
    plt.clf()

exp_labels = ["FedAvg", "CVG-CVS", "CVG-RS", "RG-CVS", "RG-RS"]
data_file_marks = [ "", "", "", "", ""]
data_dirs = [ 
    "../../exp_data/fedavg/", 
    "../../exp_data/grouping/cvg_cvs/", 
    "../../exp_data/grouping/cvg_rs/", 
    "../../exp_data/grouping/rg_cvs/", 
    "../../exp_data/grouping/rg_rs/"
    ]
accu_data_files = [ directory + "accu" + data_file_marks[i] for i, directory in enumerate(data_dirs) ]
loss_data_files = [ directory + "loss" + data_file_marks[i] for i, directory in enumerate(data_dirs) ]
cost_data_files = [ directory + "cost" + data_file_marks[i] for i, directory in enumerate(data_dirs) ]

accu_datas = [ read_last_line_data(path) for path in accu_data_files ]
loss_datas = [ read_last_line_data(path) for path in loss_data_files ]
cost_datas = [ read_last_line_data(path) for path in cost_data_files ]


draw_accu_by_round(accu_datas, exp_labels)
draw_loss_by_round(loss_datas, exp_labels)
draw_accu_by_cost(accu_datas, cost_datas, exp_labels)
print("Done")