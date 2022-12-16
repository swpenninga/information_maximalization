import argparse
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import CVAE
import utilities_network
import sampling_algorithm


def load_data(path, batch_size, train=False):
    data_loader = torch.utils.data.DataLoader(
        datasets.MNIST(path, download=False, train=train, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])), batch_size=batch_size, shuffle=True)
    return data_loader


def main(args):
    data_loc = 'D://5LSL0-Datasets'
    train_loader = load_data(args.data_path, batch_size=args.batch_size, train=True)
    test_loader = load_data(args.data_path, batch_size=args.batch_size, train=False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Running on ' + str(device))

    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    cvae = CVAE.CVAE(args.num_z)
    if args.train:
        cvae = utilities_network.trainnet(cvae, train_loader, test_loader, device, args)
    else:
        cvae.load_state_dict(torch.load("models/model_z"+str(args.num_z)))

    if args.plot:
        plot_loader = load_data(args.data_path, batch_size=5000, train=False)
        utilities_network.plotting(cvae, plot_loader, args.num_z)
        if args.num_z == 3:
            utilities_network.plot_zspace(cvae, number=5, num_samples=64)

    if args.sampling_algorithm:
        mh_loader = load_data(args.data_path, batch_size=1, train=False)
        _, (image, label) = next(enumerate(mh_loader))

        sampler = sampling_algorithm.SA(cvae.decoder, device, args)
        sampler.algorithm(image, label)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=bool, default=False)
    parser.add_argument("--plot", type=bool, default=False)
    parser.add_argument("--sampling_algorithm", type=bool, default=True)

    parser.add_argument("--seed", type=int, default=766)
    parser.add_argument("--epochs", type=int, default=55)
    parser.add_argument("--num_z", type=int, default=15)
    parser.add_argument("--mh_steps", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--data_path", type=str, default='D://MNIST')

    arguments = parser.parse_args()
    main(arguments)




