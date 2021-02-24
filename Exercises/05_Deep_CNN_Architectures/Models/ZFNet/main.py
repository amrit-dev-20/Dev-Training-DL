"""
Training & Testing of LeNet Neural Network Model.
"""
import time as time

import torch
import torch.nn as nn
import torch.optim as optim

import dataset
from ZFNet import ZFNet


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Initializing the ZFNet Object
    net = ZFNet().to(device)
    # Initializing the Cross Entropy Loss Function
    loss_fn = nn.CrossEntropyLoss()
    # Initializing the Adam Optimizer
    opt = optim.Adam(net.parameters())

    start_time = time.time()
    # Training the Model
    net.fit(dataset.trainloader, dataset.testloader, max_epochs=16, opt=opt, loss_fn=loss_fn)
    duration = time.time() - start_time
    print("Duration of Model: {:.2f} secs".format(duration))

if __name__ == "__main__":
    main()
