"""
Testing Models on CIDAR10 Dataset
"""
import argparse
import dataset
from Models.ZFNet.ZFNet import ZFNet
from trainclassifier import ClassifierTrain


def main():
    model = ZFNet()
    training = ClassifierTrain(model, dataset.trainloader, dataset.testloader, learning_rate=0.001)
    training.train(5)

if __name__ == "__main__":
    main()
