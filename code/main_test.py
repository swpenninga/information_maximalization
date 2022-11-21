import argparse
import torch
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime

import CVAE_test
import MNIST_dataloader


def main(args):

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data_loc = 'D://5LSL0-Datasets'
    batch_size = 128
    train_loader, test_loader = MNIST_dataloader.create_dataloaders(data_loc, batch_size)
    num_z = 2
    cvae = CVAE_test.AE(num_z)
    optimizer = torch.optim.Adam(cvae.parameters(), lr=args.learning_rate)
    if args.train:
        cvae = trainnet(cvae, train_loader, test_loader, optimizer, num_z)
    else:
        cvae.load_state_dict(torch.load("models/model170516"))

    if args.plot:
        plotting(cvae, test_loader.dataset.Clean_Images + 1, test_loader.dataset.Labels, num_z)


def plotting(cvae, data, labels, num_z, fig_size=(10, 10)):
    cvae.eval()
    xhat, x_z, x_c, z, c = cvae(data, labels/9)

    z = z.detach().numpy()
    fig = plt.figure(figsize=fig_size)
    if num_z == 3:
        ax = fig.add_subplot(projection='3d')
        ax.scatter(z[:, 0], z[:, 1], z[:, 2], c=labels, cmap='tab10')
    else:
        ax = fig.add_subplot()
        ax.scatter(z[:, 0], z[:, 1], c=labels, cmap='tab10')
    fig.suptitle('distribution of latent variable z')
    fig.show()

    c = c.detach().numpy()
    fig = plt.figure(figsize=fig_size)
    if num_z == 3:
        ax = fig.add_subplot(projection='3d')
        ax.scatter(c[:, 0], c[:, 1], c[:, 2], c=labels, cmap='tab10')
    else:
        ax = fig.add_subplot()
        ax.scatter(c[:, 0], c[:, 1], c=labels, cmap='tab10')
    fig.suptitle('distribution of latent class variable c')
    fig.show()

    return


def trainnet(cvae, train_loader, test_loader, optimizer, num_z):
    criterion = nn.MSELoss()
    for epoch in range(args.epochs):
        totalloss = 0.0
        print(f"\nTraining Epoch {epoch + 1}:")
        cvae.train()

        for idx, (x, _, labels_x) in enumerate(tqdm(train_loader)):
            optimizer.zero_grad()
            xhat, x_z, x_c, z, c = cvae(x+1, labels_x/9)
            loss, MSE, KLD = loss_fn(xhat, x+1, x_z, x_c, criterion)
            loss.backward()
            optimizer.step()
            totalloss += loss.item()
            if idx % 100 == 99:
                print(f"loss:[{totalloss/100:.2e}] MSE:[{MSE:.2e}] KLD:[{KLD.item():.2e}]")
                totalloss = 0.0

        cvae.eval()
        totalvalloss = 0.0
        for idx, (x, _, labels_x) in enumerate(test_loader):
            xhat, x_z, x_c, z, c = cvae(x+1, labels_x/9)
            loss, MSE, KLD = loss_fn(xhat, x+1, x_z, x_c, criterion)
            totalvalloss += loss.item()
            if idx % 50 == 49:
                print(f"loss:[{totalvalloss/50:.2e}] MSE:[{MSE:.2e}] KLD:[{KLD.item():.2e}]")
                totalvalloss = 0.0
        cvae.train()

    save_model(cvae, num_z)

    return cvae


def save_model(cvae, num_z):
    now = datetime.now()
    current_time = now.strftime("%H%M%S")
    torch.save(cvae.state_dict(), "models/model" + current_time + 'test' + str(num_z))
    print("Saved model weights as model" + current_time + ".")
    return


def loss_fn(x_hat, x, x_z, x_c, criterion, beta=1e-6):
    MSE = criterion(x_hat, x)
    KLD_z = torch.zeros(1)
    KLD_c = torch.zeros(1)
    for i in range(int(x_z.size()[1]/2)):
        KLD_z += torch.sum(torch.square(x_z[:, i]) - 2*x_z[:, i+int(x_z.size()[1]/2)] + torch.square(torch.exp(x_z[:, i+int(x_z.size()[1]/2)])) - 1)
        KLD_c += torch.sum(torch.square(x_c[:, i]) - 2*x_c[:, i+int(x_c.size()[1]/2)] + torch.square(torch.exp(x_c[:, i+int(x_c.size()[1]/2)])) - 1)
    KLD = KLD_z + KLD_c
    return (MSE + beta*KLD)/x.size()[0], MSE/x.size()[0], beta*KLD/x.size()[0]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1523423)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--plot", type=bool, default=True)
    parser.add_argument("--train", type=bool, default=True)

    args = parser.parse_args()

    main(args)




