"""
Implementation of the LeNet Architecture
"""
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.cnn_model = nn.Sequential(
            nn.Conv2d(3, 6, 5),
            nn.Tanh(),
            nn.AvgPool2d(2, stride=2),
            nn.Conv2d(6, 16, 5),
            nn.Tanh(),
            nn.AvgPool2d(2, stride=2)
        )
        self.fc_model = nn.Sequential(
            nn.Linear(400, 120),
            nn.Tanh(),
            nn.Linear(120, 84),
            nn.Tanh(),
            nn.Linear(84, 10)
        )

    def forward(self, x):
        x = self.cnn_model(x)
        x = x.view(x.size(0), - 1)
        x = self.fc_model(x)
        return x

    def evaluation(self, dataloader):
        total, correct = 0, 0
        for data in dataloader:
            inputs, labels = data
            outputs = self(inputs)
            _, pred = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (pred == labels).sum().item()
        return 100 * correct / total

    def fit(self, train_loader, test_loader, max_epochs, opt, loss_fn):
        loss_arr = []
        loss_epoch_arr = []

        for epoch in range(max_epochs):

            for i, data in enumerate(train_loader, 0):

                inputs, labels = data
                opt.zero_grad()
                outputs = self(inputs)
                loss = loss_fn(outputs, labels)
                loss.backward()
                opt.step()

                loss_arr.append(loss.item())
            loss_epoch_arr.append(loss.item())
            print('Epoch: %d/%d, Test acc: %0.2f, Train acc: %0.2f' % (epoch, max_epochs, self.evaluation(test_loader), self.evaluation(train_loader)))
        plt.plot(loss_epoch_arr)
        plt.show()
