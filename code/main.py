import argparse
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import concurrent.futures
from datetime import datetime

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
    data_loc = 'C:/MNIST'
    train_loader = load_data(args.data_path, batch_size=args.batch_size, train=True)
    test_loader = load_data(args.data_path, batch_size=args.batch_size, train=False)
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
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
        now = datetime.now()
        print('Time of start: ')
        print(now)

        mh_loader = load_data(args.data_path, batch_size=1, train=False)
        sampler = sampling_algorithm.SA(cvae.decoder, device, args)

        data_list = []
        for i in range(args.num_images):
            _, (image, label) = next(enumerate(mh_loader))
            data_tensor = torch.empty(1, 2, 28, 28)
            data_tensor[0, 0, :, :] = image
            data_tensor[0, 1, 0, 0] = label
            data_list.append(data_tensor)

        with concurrent.futures.ProcessPoolExecutor() as executor:
            results = executor.map(sampler.algorithm, data_list)
        now = datetime.now()
        current_time = now.strftime("%H%M%S")
        torch.save(list(results), 'data/run' + current_time + '.pt')
        print('Time of finish:')
        print(now)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default='D://MNIST')

    parser.add_argument("--train", type=bool, default=False)
    parser.add_argument("--plot", type=bool, default=False)
    parser.add_argument("--sampling_algorithm", type=bool, default=True)

    parser.add_argument("--seed", type=int, default=766)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--num_z", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=64)

    parser.add_argument("--num_images", type=int, default=1)
    parser.add_argument("--mh_steps", type=int, default=75)
    parser.add_argument("--num_pixels", type=int, default=25)
    parser.add_argument("--mc_sigma", type=int, default=0.10)
    parser.add_argument("--exp_amp", type=int, default=25)
    parser.add_argument("--print_mh", type=bool, default=True)

    arguments = parser.parse_args()
    main(arguments)




