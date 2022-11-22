import argparse
import torch
import torch.nn as nn
from tqdm import tqdm


import CVAE
import MNIST_dataloader
import utilities


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
        cvae.load_state_dict(torch.load("models/model164805"))

    if args.plot:
        utilities.plotting(cvae, (train_loader.dataset.Clean_Images[0:5000,:,:,:] + 1)/2, train_loader.dataset.Labels[0:5000], num_z)


def trainnet(cvae, train_loader, test_loader, optimizer):
    criterion = nn.MSELoss()
    for epoch in range(args.epochs):
        totalloss = 0.0
        print(f"\nTraining Epoch {epoch + 1}:")

        cvae.train()
        for idx, (x, _, labels_x) in enumerate(tqdm(train_loader)):
            optimizer.zero_grad()
            xhat, x_z, x_c, z, c = cvae((x+1)/2, labels_x)
            loss, MSE, KLD = loss_fn(xhat, (x+1)/2, x_z, x_c, criterion)
            loss.backward()
            optimizer.step()
            totalloss += loss.item()
            if idx % 100 == 99:
                print(f"loss:[{totalloss/100:.2e}] MSE:[{MSE:.2e}] KLD:[{KLD.item():.2e}]")
                totalloss = 0.0

        cvae.eval()
        totalvalloss = 0.0
        for idx, (x, _, labels_x) in enumerate(test_loader):
            xhat, x_z, x_c, z, c = cvae((x+1)/2, labels_x)
            loss, MSE, KLD = loss_fn(xhat, (x+1)/2, x_z, x_c, criterion)
            totalvalloss += loss.item()
            if idx % 50 == 49:
                print(f"loss:[{totalvalloss/50:.2e}] MSE:[{MSE:.2e}] KLD:[{KLD.item():.2e}]")
                totalvalloss = 0.0
    utilities.save_model(cvae)
    return cvae


def loss_fn(x_hat, x, x_z, x_c, criterion, beta=1e-7, zeta=1e1):
    MSE = criterion(x_hat, x)
    KLD_z = torch.zeros(1)
    KLD_c = torch.zeros(1)
    for i in range(int(x_z.size()[1]/2)):
        KLD_z += torch.sum(torch.square(x_z[:, i]) - 2*x_z[:, i+int(x_z.size()[1]/2)] + torch.square(torch.exp(x_z[:, i+int(x_z.size()[1]/2)])) - 1)
        KLD_c += torch.sum(torch.square(x_c[:, i]) - 2*x_c[:, i+int(x_c.size()[1]/2)] + torch.square(torch.exp(x_c[:, i+int(x_c.size()[1]/2)])) - 1)
    KLD = KLD_z + KLD_c * zeta
    return (MSE + beta*KLD)/x.size()[0], MSE/x.size()[0], beta*KLD/x.size()[0]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1523423)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--plot", type=bool, default=True)
    parser.add_argument("--train", type=bool, default=False)

    args = parser.parse_args()

    main(args)




