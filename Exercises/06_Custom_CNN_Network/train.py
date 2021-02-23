import argparse
from team_classifier_train import TeamClassifierTrain
from dataloader import Dataloader

def main(args):
    data = Dataloader()
    trainloader, testloader = data.data_loader(args.train_filepath, args.test_filepath, args.batch_size)
    training = TeamClassifierTrain(trainloader, testloader, args.learning_rate)
    training.train(args.epochs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_filepath', type=str, help='Enter the Training Image Dataset', required=True)
    parser.add_argument('--test_filepath', type=str, help='Enter the Testing Image Dataset', required=True)
    parser.add_argument('--batch_size', type=int, help="Enter the Batch Size", required=True)
    parser.add_argument('--epochs', type=int, help="Enter the No of Epochs", required=True)
    parser.add_argument('--learning_rate', type=float, default=0.001, help="Enter the Learning Rate")
    args = parser.parse_args()
    main(args)
