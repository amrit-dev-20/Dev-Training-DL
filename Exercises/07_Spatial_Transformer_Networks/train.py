import argparse
from data.dataloader import dataset
from stn_trainer import STNTrain

def main(args):
    train_loader, test_loader = dataset()
    model_obj = STNTrain(train_loader, test_loader, learning_rate=0.001)
    model_obj.train(25, args.checkpoint_path, args.export_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_path', type=str, default='Exercises/07_Spatial_Transformer_Networks/checkpoints')
    parser.add_argument('--export_path', type=str, default='Exercises/07_Spatial_Transformer_Networks/exports')
    main(parser.parse_args())
