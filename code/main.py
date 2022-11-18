import argparse
import torch
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
    num_z = 3
    cvae = CVAE.AE(num_z)
    optimizer = torch.optim.Adam(cvae.parameters(), lr=args.learning_rate)
    if args.train:
        cvae = trainnet(cvae, train_loader, test_loader, optimizer)
    else:
        cvae.load_state_dict(torch.load("models/model170516"))

    if args.plot:
        plotting(cvae, test_loader.dataset.Clean_Images + 1, test_loader.dataset.Labels, num_z)


def plotting(cvae, data, labels, num_z, fig_size=(10, 10)):
    cvae.eval()
    xhat, h, z = cvae(data)
    z = z.detach().numpy()
    fig = plt.figure(figsize=fig_size)
    if num_z == 3:
        ax = fig.add_subplot(projection='3d')
        ax.scatter(z[:, 0], z[:, 1], z[:, 2], c=labels, cmap='tab10')
    else:
        ax = fig.add_subplot()
        ax.scatter(z[:, 0], z[:, 1], c=labels, cmap='tab10')
    # ax.colorbar()
    fig.show()


def trainnet(cvae, train_loader, test_loader, optimizer):
    criterion = nn.MSELoss()
    for epoch in range(args.epochs):

        totalloss = 0.0

        print(f"\nTraining Epoch {epoch + 1}:")
        cvae.train()

        for idx, (x, _, labels_x) in enumerate(tqdm(train_loader)):
            optimizer.zero_grad()
            xhat, h, z = cvae(x+1)
            loss, MSE, KLD = loss_fn(xhat, x+1, h, criterion)
            loss.backward()
            optimizer.step()
            totalloss += loss.item()
            if idx % 100 == 99:
                print(f"loss:[{totalloss/100:.2e}] MSE:[{MSE:.2e}] KLD:[{KLD.item():.2e}]")
                totalloss = 0.0

        cvae.eval()
        totalvalloss = 0.0
        for idx, (x, _, _) in enumerate(test_loader):
            xhat, h, z = cvae(x)
            loss, MSE, KLD = loss_fn(xhat, x+1, h, criterion)
            totalvalloss += loss.item()
            if idx % 50 == 49:
                print(f"loss:[{totalvalloss/50:.2e}] MSE:[{MSE:.2e}] KLD:[{KLD.item():.2e}]")
                totalvalloss = 0.0
        cvae.train()

    save_model(cvae)

    return cvae

def save_model(cvae):
    now = datetime.now()
    current_time = now.strftime("%H%M%S")
    torch.save(cvae.state_dict(), "models/model" + current_time)
    print("Saved model weights as model" + current_time + ".")
    return


# 1e-6 works for z_dim=2
# 1e-5 works for z_dim=3


def loss_fn(x_hat, x, h, criterion, beta=1e-5):
    # BCE = torch.nn.functional.binary_cross_entropy(x_hat, x, reduction='sum')
    MSE = criterion(x_hat, x)
    KLD = torch.zeros(1)
    for i in range(int(h.size()[1]/2)):
        KLD += torch.sum(torch.square(h[:, i]) - 2*h[:, i+int(h.size()[1]/2)] + torch.square(torch.exp(h[:, i+int(h.size()[1]/2)])) - 1)
        # KLD += -0.5 * torch.sum(1 + h[:, i] - h[:, i+int(h.size()[1]/2)].pow(2) - h[:, i].exp())
    return (MSE + beta*KLD)/x.size()[0], MSE/x.size()[0], beta*KLD/x.size()[0]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1523423)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--plot", type=bool, default=True)
    parser.add_argument("--train", type=bool, default=True)

    args = parser.parse_args()

    main(args)




