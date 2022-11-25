import torch
from torch import nn
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
from datetime import datetime


def loss_fn(x_hat, x, x_z, x_c, criterion, device, beta=2e-7, zeta=0.5):
    MSE = criterion(x_hat, x)
    KLD_z = torch.zeros(1, device=device)
    KLD_c = torch.zeros(1, device=device)
    for i in range(int(x_z.size()[1]/2)):
        KLD_z += torch.sum(torch.square(x_z[:, i]) - 2*x_z[:, i+int(x_z.size()[1]/2)] +
                           torch.square(torch.exp(x_z[:, i+int(x_z.size()[1]/2)])) - 1)
        KLD_c += torch.sum(torch.square(x_c[:, i]) - 2*x_c[:, i+int(x_c.size()[1]/2)] +
                           torch.square(torch.exp(x_c[:, i+int(x_c.size()[1]/2)])) - 1)
    KLD = KLD_z + KLD_c * zeta
    return (MSE + beta*KLD)/x.size()[0], MSE/x.size()[0], beta*KLD/x.size()[0]


def trainnet(cvae, train_loader, test_loader, device, args):
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    print('Training on ' + str(device))

    optimizer = torch.optim.Adam(cvae.parameters(), lr=args.learning_rate)
    criterion = nn.MSELoss()
    cvae.to(device)

    for epoch in range(args.epochs):
        totalloss = 0.0
        print(f"\nTraining Epoch {epoch + 1}:")
        cvae.train()
        for idx, (x, _, labels_x) in enumerate(tqdm(train_loader)):
            x = x.to(device)
            labels_x = labels_x.to(device)
            labels_x = torch.nn.functional.one_hot(labels_x, num_classes=10)
            x = (x + 1) / 2

            optimizer.zero_grad()
            xhat, x_z, x_c, z, c = cvae(x, labels_x, device)
            loss, MSE, KLD = loss_fn(xhat, (x + 1) / 2, x_z, x_c, criterion, device)
            loss.backward()
            optimizer.step()
            totalloss += loss.item()
            if idx % 100 == 99:
                print(f"loss:[{totalloss / 100:.2e}] MSE:[{MSE:.2e}] KLD:[{KLD.item():.2e}]")
                totalloss = 0.0

        cvae.eval()
        totalvalloss = 0.0
        with torch.no_grad():
            for idx, (x, _, labels_x) in enumerate(test_loader):
                x = x.to(device)
                labels_x = labels_x.to(device)
                labels_x = torch.nn.functional.one_hot(labels_x, num_classes=10)
                x = (x + 1) / 2

                xhat, x_z, x_c, z, c = cvae(x, labels_x, device)
                loss, MSE, KLD = loss_fn(xhat, (x + 1) / 2, x_z, x_c, criterion, device)
                totalvalloss += loss.item()
                if idx % 50 == 49:
                    print(f"valloss:[{totalvalloss / 50:.2e}] MSE:[{MSE:.2e}] KLD:[{KLD.item():.2e}]")
                    totalvalloss = 0.0
    # Make loss plot function from args.plot
    cvae.to(torch.device('cpu'))
    save_model(cvae, args.num_z)
    return cvae


def save_model(cvae, num_z):
    now = datetime.now()
    current_time = now.strftime("%H%M%S")
    torch.save(cvae.state_dict(), "models/model" + current_time + 'z' + str(num_z))
    print("Saved model weights as model" + current_time + 'z' + str(num_z) + ".")
    return


def plotting(cvae, data, labels, num_z, fig_size=(10, 10)):
    cvae.eval()
    device = torch.device('cpu')
    xhat, x_z, x_c, z, c = cvae(data, torch.nn.functional.one_hot(labels, num_classes=10), device)
    z = z.detach().numpy()
    c = c.detach().numpy()
    if num_z == 3 or num_z == 2:
        fig = plt.figure(figsize=fig_size)
        if num_z == 3:
            ax = fig.add_subplot(projection='3d')
            scatter = ax.scatter(z[:, 0], z[:, 1], z[:, 2], c=labels, cmap='tab10')
        else:
            ax = fig.add_subplot()
            scatter = ax.scatter(z[:, 0], z[:, 1], c=labels, cmap='tab10')
        legend = ax.legend(*scatter.legend_elements(), loc="center left", prop={'size': 16})
        ax.add_artist(legend)
        fig.suptitle('latent space of z')
        fig.show()

        fig = plt.figure(figsize=fig_size)
        if num_z == 3:
            ax = fig.add_subplot(projection='3d')
            scatter = ax.scatter(c[:, 0], c[:, 1], c[:, 2], c=labels, cmap='tab10')
            # ax.view_init(elev=-45, azim=-20)
            # ax.view_init(elev=-70, azim=-60)
        else:
            ax = fig.add_subplot()
            scatter = ax.scatter(c[:, 0], c[:, 1], c=labels, cmap='tab10')
        legend = ax.legend(*scatter.legend_elements(), loc="center left", prop={'size': 16})
        ax.add_artist(legend)
        fig.suptitle('latent space of c')
        fig.show()

    x_clean_example = data[0:10, :, :, :]
    output_example = xhat[0:10, :, :, :]
    output_example = output_example.detach().numpy()

    plt.figure(figsize=(12, 3))
    for i in range(10):
        plt.subplot(2, 10, i + 1)
        plt.imshow(x_clean_example[i, 0, :, :], cmap='gray')
        plt.xticks([])
        plt.yticks([])

        plt.subplot(2, 10, i + 11)
        plt.imshow(output_example[i, 0, :, :], cmap='gray')
        plt.xticks([])
        plt.yticks([])
    plt.tight_layout()
    plt.suptitle('output of the auto-encoder(num_z='+str(num_z)+') with respect to input numbers' + str(labels[10:20].numpy()))
    plt.show()
    return


def plot_zspace(cvae, num_samples=64, pos=(1.5, -1, 1)):
    points_per_dim = torch.pow(torch.tensor(num_samples), 1/3)
    x = torch.linspace(-1, 1, int(points_per_dim.item()))
    plt.figure(figsize=(12, 12))
    for i in range(num_samples):
        latent_vars = torch.tensor([x[int(np.floor(i / points_per_dim.item()**2) % points_per_dim.item()**2)],
                                    x[int(np.floor(i/points_per_dim.item()) % points_per_dim.item())],
                                    x[i % int(points_per_dim.item())],
                                    pos[0], pos[1], pos[2]])
        out = cvae.decoder(torch.unsqueeze(latent_vars, 0))
        plt.subplot(int(np.sqrt(num_samples)), int(np.sqrt(num_samples)), i+1)
        plt.imshow(np.squeeze(out.detach().numpy()))
        plt.xticks([])
        plt.yticks([])

    plt.tight_layout()
    plt.show()
    return

