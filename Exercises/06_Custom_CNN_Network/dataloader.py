"""
Dataloader for Binary Classification.
"""
import argparse
import torch
import torchvision.transforms as transforms
from torchvision import datasets

class Dataloader:
    def __init__(self):
        self.train_images = None
        self.test_images = None
        self.data_transform = None

    def data_loader(self, train_filepath, test_filepath, batch_size, resize=(128, 256)):
        self.data_transform_train = transforms.Compose([
            transforms.RandomRotation(40),
            transforms.Resize(resize),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.data_transform_test = transforms.Compose([
            transforms.Resize(resize),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.train_images = datasets.ImageFolder(train_filepath, transform=self.data_transform_train)
        trainloader = torch.utils.data.DataLoader(self.train_images, batch_size=batch_size, shuffle=True, num_workers=2)

        self.test_images = datasets.ImageFolder(test_filepath, transform=self.data_transform_test)
        testloader = torch.utils.data.DataLoader(self.test_images, batch_size=batch_size, shuffle=False, num_workers=2)

        return trainloader, testloader

def main(args):
    data = Dataloader()
    trainloader, testloader = data.data_loader(args.train_filepath, args.test_filepath, args.batch_size)
    dataiter = iter(trainloader)
    images, labels = dataiter.next()
    print(images.shape)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_filepath', type=str, help='Enter the Training Image Dataset', required=True)
    parser.add_argument('--test_filepath', type=str, help='Enter the Testing Image Dataset', required=True)
    parser.add_argument('--batch_size', type=int, help="Enter the Batch Size", required=True)
    args = parser.parse_args()
    main(args)
