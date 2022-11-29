import argparse
import torch

import CVAE
import MNIST_dataloader
import utilities_network
import MH_algorithm


def main(args):
    data_loc = 'D://5LSL0-Datasets'
    train_loader, test_loader = MNIST_dataloader.create_dataloaders(data_loc, args.batch_size)
    cvae = CVAE.AE(args.num_z)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.train:
        cvae = utilities_network.trainnet(cvae, train_loader, test_loader, device, args)
    else:
        cvae.load_state_dict(torch.load("models/model155149z3"))

    if args.MH:
        mh_sampler = MH_algorithm.MH(cvae, device, args)
        memory = torch.Tensor()
        init_sample = torch.zeros((1, args.num_z*2))
        for _ in range(args.mh_steps):
            sample_set, log_posterior = mh_sampler.sample(memory=memory, init_sample=init_sample)
            taken_action = MH_algorithm.action_num(cvae, sample_set, memory, args.image_size, device)
            memory = torch.cat((memory, taken_action))
            init_sample = sample_set[torch.argmax(log_posterior)].unsqueeze(0)

    if args.plot:
        utilities_network.plotting(cvae, (train_loader.dataset.Clean_Images[0:6000, :, :, :] + 1)/2,
                                   train_loader.dataset.Labels[0:6000], args.num_z)
        utilities_network.plot_zspace(cvae, num_samples=64, pos=(-1, -1, 1))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=122355534)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--image_size", type=int, default=32*32)
    parser.add_argument("--mh_steps", type=int, default=50)
    parser.add_argument("--num_z", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--plot", type=bool, default=False)
    parser.add_argument("--train", type=bool, default=False)
    parser.add_argument("--MH", type=bool, default=True)

    arguments = parser.parse_args()
    main(arguments)




