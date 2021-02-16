"""
Training & Testing of LeNet Neural Network Model.
"""
from LeNet import LeNet
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim


def main():
    # Loading Training and Testing Dataset from CIFAR10
    batch_size = 128
    train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transforms.ToTensor())
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)

    test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transforms.ToTensor())
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)
    # Classes
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # Getting Images with the dataset
    dataiter = iter(train_loader)
    images, labels = dataiter.next()

    # Initializing the LeNet Object
    net = LeNet()
    # Initializing the Cross Entropy Loss Function
    loss_fn = nn.CrossEntropyLoss()
    # Initializing the Adam Optimizer
    opt = optim.Adam(net.parameters())

    # Training the Model
    net.fit(train_loader, test_loader, max_epochs=16, opt=opt, loss_fn=loss_fn)

if __name__ == "__main__":
    main()
