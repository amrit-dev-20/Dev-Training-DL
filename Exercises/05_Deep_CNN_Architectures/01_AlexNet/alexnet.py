"""
Implementation of AlexNet Model
"""
import os

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms


class Alexnet:
    def __init__(self):
        self.model = torch.hub.load('pytorch/vision:v0.6.0', 'alexnet', pretrained=True)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.running_loss = 0.0
        self.path = 'Exercises/05_Deep_CNN_Architectures/01_AlexNet/Loss_Graph/'

    def eval(self):
        return self.model.eval()

    def update_classifier(self, net_layer, neuron_input, neuron_output):
        self.model.classifier[net_layer] = nn.Linear(neuron_input, neuron_output)

    def fit(self, trainloader, testloader, epochs, optimizer, criterion):
        loss_arr = []
        loss_epoch_arr = []
        for i in range(epochs):  # loop over the dataset multiple times
            self.running_loss = 0.0
            for j, data in enumerate(trainloader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data[0].to(self.device), data[1].to(self.device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                output = self.model(inputs)
                loss = criterion(output, labels)
                loss.backward()
                optimizer.step()

                loss_arr.append(loss.item())
                # print statistics
                self.running_loss += loss.item()
                if j % 2000 == 1999:    # print every 2000 mini-batches
                    print('[%d, %5d] loss: %.3f' % (i + 1, j + 1, self.running_loss / 2000))
                    self.running_loss = 0.0
            loss_epoch_arr.append(loss.item())
        plt.plot(loss_epoch_arr)
        plt.savefig(os.path.join(self.path, 'AlexNet_Loss_Graph.png'))
        plt.show()
        print('Finished Training of AlexNet')

    def evaluation(self):
        pass
