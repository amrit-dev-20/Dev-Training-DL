"""
Implementation of ZFNet using Pytorch
"""
import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


class ZFNet(nn.Module):
    def __init__(self):
        super(ZFNet, self).__init__()
        self.cnn_model = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=7, stride=(4, 4)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(96, 256, kernel_size=5, stride=(1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(256, 512, kernel_size=3, stride=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 1024, kernel_size=3, stride=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 512, kernel_size=3, stride=(1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.fc_model = nn.Sequential(
            nn.Linear(2048, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(4096, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(1024, 10)
        )
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.path = ''

    def forward(self, x):
        x = self.cnn_model(x)
        x = x.view(x.size(0), -1)
        x = self.fc_model(x)
        return x

    def evaluation(self, dataloader):
        total, correct = 0, 0
        for data in dataloader:
            inputs, labels = data
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            outputs = self(inputs)
            _, pred = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (pred == labels).sum().item()
        return 100 * correct / total

    def fit(self, train_loader, test_loader, max_epochs, opt, loss_fn):
        loss_arr = []
        loss_epoch_arr = []

        for epoch in range(max_epochs):
            self.running_loss = 0.0
            for i, data in enumerate(train_loader, 0):

                inputs, labels = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                opt.zero_grad()
                outputs = self(inputs)
                loss = loss_fn(outputs, labels)
                loss.backward()
                opt.step()

                loss_arr.append(loss.item())
                self.running_loss += loss.item()
                if i % 2000 == 1999:    # print every 2000 mini-batches
                    print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, self.running_loss / 2000))
                    self.running_loss = 0.0
            loss_epoch_arr.append(loss.item())
            print('Epoch: %d/%d, Test acc: %0.2f, Train acc: %0.2f' % (epoch, max_epochs, self.evaluation(test_loader), self.evaluation(train_loader)))
        plt.plot(loss_epoch_arr)
        plt.savefig(os.path.join(self.path, 'LeNet_Loss_Graph.png'))
        plt.show()
