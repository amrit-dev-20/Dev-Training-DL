"""
Testing of Alexnet on CIDAR10 Dataset
"""
import time as time

import torch
import torch.nn as nn
import torch.optim as optim

import dataset
from alexnet import Alexnet


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    alexnet = Alexnet()
    # print(alexnet.eval())
    alexnet.update_classifier(4, 4096, 1024)
    alexnet.update_classifier(6, 1024, 10)
    print(alexnet.eval())

    alexnet.model.to(device)
    # Loss
    criterion = nn.CrossEntropyLoss()

    # Optimizer(SGD)
    optimizer = optim.Adam(alexnet.model.parameters(), lr=0.0001)

    # Training the Alexnet Model
    alexnet.fit(trainloader=dataset.trainloader, testloader=dataset.testloader, epochs=5, optimizer=optimizer, criterion=criterion)


if __name__ == "__main__":
    main()
