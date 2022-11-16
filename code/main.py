import argparse
import torch
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime

import CVAE
import MNIST_dataloader


def main(args):

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data_loc = 'D://5LSL0-Datasets'
    batch_size = 64
    train_loader, test_loader = MNIST_dataloader.create_dataloaders(data_loc, batch_size)

    cvae = CVAE.AE()
    optimizer = torch.optim.Adam(cvae.parameters(), lr=args.learning_rate)
    if args.train:
        cvae = trainnet(cvae, train_loader, test_loader, optimizer)
    else:
        cvae.load_state_dict(torch.load("models/model153616"))

    if args.plot:
        plotting(cvae, test_loader.dataset.Clean_Images, test_loader.dataset.Labels)


def plotting(cvae, data, labels, fig_size=(10, 10)):
    cvae.eval()
    xhat, z = cvae(data)
    z = z.detach().numpy()

    plt.figure(figsize=fig_size)
    plt.scatter(z[:, 0, 0, 0], z[:, 0, 1, 0], c=labels, cmap='tab10')
    plt.colorbar()
    plt.show()


def trainnet(cvae, train_loader, test_loader, optimizer):
    for epoch in range(args.epochs):

        totalloss = 0.0
        totalvalloss = 0.0
        print(f"\nTraining Epoch {epoch + 1}:")
        cvae.train()

        for idx, (x, _, labels_x) in enumerate(tqdm(train_loader)):
            optimizer.zero_grad()
            xhat, z = cvae(x)
            loss = loss_fn(xhat, x, z[:, 0, 0, 0], z[:, 0, 1, 0])
            loss.backward()
            optimizer.step()
            totalloss += loss.item()
            if idx % 100 == 99:
                print(f"loss:[{totalloss/100:.4f}]")
                totalloss = 0.0

        cvae.eval()
        for idx, (x, _, _) in enumerate(test_loader):
            xhat, z = cvae(x)
            loss = loss_fn(xhat, x, z[:, 0, 0, 0], z[:, 0, 1, 0])
            totalvalloss += loss.item()
        cvae.train()
        print(f'val_loss:[{totalvalloss / len(test_loader):.4f}]')

    now = datetime.now()
    current_time = now.strftime("%H%M%S")
    torch.save(cvae.state_dict(), "models/model" + current_time)
    print("Saved model weights as model" + current_time + ".")

    return cvae

def loss_fn(x_hat, x, mean, log_var):
    # BCE = torch.nn.functional.binary_cross_entropy(x_hat, x, reduction='sum')
    criterion = nn.MSELoss()
    MSE = criterion(x_hat, x)
    KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
    return (MSE + 0.001*KLD) / x.size(0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--plot", type=bool, default=True)
    parser.add_argument("--train", type=bool, default=False)

    args = parser.parse_args()

    main(args)




