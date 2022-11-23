import argparse
import torch

import CVAE
import MNIST_dataloader
import utilities_network


def main(args):
    data_loc = 'D://5LSL0-Datasets'
    train_loader, test_loader = MNIST_dataloader.create_dataloaders(data_loc, args.batch_size)
    cvae = CVAE.AE(args.num_z)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.train:
        cvae = utilities_network.trainnet(cvae, train_loader, test_loader, device, args)
    else:
        cvae.load_state_dict(torch.load("models/model141233z3"))

    if args.plot:
        utilities_network.plotting(cvae, (train_loader.dataset.Clean_Images[0:5000, :, :, :] + 1)/2,
                                   train_loader.dataset.Labels[0:5000], args.num_z)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=123124234534)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--num_z", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--plot", type=bool, default=True)
    parser.add_argument("--train", type=bool, default=False)

    args = parser.parse_args()
    main(args)




