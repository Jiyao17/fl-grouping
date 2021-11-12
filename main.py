
# code source:
# FashionMNIST: pytorch tutorial quickstart
# SpeechCommand: pytorch tutorial audio
# TextClassification: pytorch tutorial text

from multiprocessing import set_start_method

from utils.task import Task, TaskConfig, TaskCIFAR
from utils.models import CIFAR

import torch


if __name__ == "__main__":
    # prepare for multi-proccessing
    # if get_start_method(False) != "spawn":
    # set_start_method("spawn")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    trainset, testset = TaskCIFAR.load_dataset("./data/")

    train_config = TaskConfig(1, 32, 0.1, device)
    test_config = TaskConfig(1, 256, 0, device)

    cifar_train = TaskCIFAR(trainset, train_config)
    trainde_model = cifar_train.get_model()
    cifar_test = TaskCIFAR(testset, test_config)

    for i in range(5):
        cifar_train.train_model()

        cifar_test.set_model(cifar_train.get_model())
        acc, loss = cifar_test.test_model()
        # acc, loss = cifar_train.test_model()
        print("accuracy: " + str(acc) + ", loss: " + str(loss))




