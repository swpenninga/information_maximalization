import argparse
import torch

import CVAE
import MNIST_dataloader
import utilities_network
import sampling_algorithm


def main(args):
    data_loc = 'D://5LSL0-Datasets'
    train_loader, test_loader = MNIST_dataloader.create_dataloaders(data_loc, args.batch_size)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    if args.conditional:
        cvae = CVAE.CVAE(args.num_z)
    else:
        cvae = CVAE.AE(args.num_z)
    classifier = CVAE.Classifier()

    if args.train:
        cvae = utilities_network.trainnet(cvae, train_loader, test_loader, device, args)
        if ~args.conditional:
            classifier = utilities_network.train_classifier(classifier, train_loader, test_loader, device, args)
    else:
        if args.conditional:
            cvae.load_state_dict(torch.load("models/model122023z3cond"))
        else:
            cvae.load_state_dict(torch.load("models/model124707z3"))
            classifier.load_state_dict(torch.load("models/model_classifier125659"))

    if args.plot:
        utilities_network.plotting(cvae, train_loader.dataset.Clean_Images[0:6000, :, :, :],
                                   train_loader.dataset.Labels[0:6000], args.num_z)
        utilities_network.plot_zspace(cvae, args.conditional, num_samples=64)

    if args.sampling_algorithm:
        sampler = sampling_algorithm.SA(cvae.decoder, device, args)
        mh_loader, _ = MNIST_dataloader.create_dataloaders(data_loc, batch_size=1)

        idx, (image, _, label) = next(enumerate(mh_loader))
        image = image.squeeze()
        image = image.to(device)
        sampler.sample(image)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1225243534)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--image_size", type=int, default=32*32)
    parser.add_argument("--mh_steps", type=int, default=100)
    parser.add_argument("--num_z", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=0.001)

    parser.add_argument("--conditional", type=bool, default=False)

    parser.add_argument("--plot", type=bool, default=False)
    parser.add_argument("--train", type=bool, default=False)
    parser.add_argument("--sampling_algorithm", type=bool, default=False)

    arguments = parser.parse_args()
    main(arguments)




