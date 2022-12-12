import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
from datetime import datetime


def loss_fn(x_hat, x, z, criterion, beta=1e-6):
    MSE = criterion(x_hat, x)
    KLD = 0
    for i in range(int(z.size()[1]/2)):
        KLD += torch.sum(torch.square(z[:, i]) - 2*z[:, i+int(z.size()[1]/2)] +
                         torch.square(torch.exp(z[:, i+int(z.size()[1]/2)])) - 1)
    return (MSE + beta*KLD)/z.size()[0], MSE/z.size()[0], beta*KLD/z.size()[0]


def trainnet(cvae, train_loader, test_loader, device, args):
    print('Training on ' + str(device))

    optimizer = torch.optim.Adam(cvae.parameters(), lr=0.001)
    criterion = nn.MSELoss(reduction='mean')
    cvae.to(device)
    cvae.train()
    for epoch in range(args.epochs):
        totalloss = 0.0
        print(f"\nTraining Epoch {epoch + 1}:")
        print('Train:')
        for idx, (x, labels_x) in enumerate(tqdm(train_loader)):
            x = x.to(device)
            labels_x = labels_x.to(device)
            labels_x = F.one_hot(labels_x, num_classes=10)

            optimizer.zero_grad()
            xhat, z = cvae(x, labels_x, device)
            loss, MSE, KLD = loss_fn(xhat, x, z, criterion)
            loss.backward()
            optimizer.step()
            totalloss += loss.item()
            if idx % 100 == 99:
                print(f"loss:[{totalloss / 100:.2e}] MSE:[{MSE:.2e}] KLD:[{KLD.item():.2e}]")
                totalloss = 0.0

        cvae.eval()
        print('Eval:')
        totalvalloss = 0.0
        for idx, (x, labels_x) in enumerate(tqdm(test_loader)):
            x = x.to(device)
            labels_x = labels_x.to(device)
            labels_x = F.one_hot(labels_x, num_classes=10)

            optimizer.zero_grad()
            xhat, z = cvae(x, labels_x, device)
            loss, MSE, KLD = loss_fn(xhat, x, z, criterion)
            loss.backward()
            optimizer.step()
            totalvalloss += loss.item()
            if idx % 50 == 49:
                print(f"valloss:[{totalvalloss / 50:.2e}] valMSE:[{MSE:.2e}] valKLD:[{KLD.item():.2e}]")
                totalvalloss = 0.0

    cvae.to(torch.device('cpu'))
    save_model(cvae, args.num_z)
    return cvae


def save_model(cvae, num_z):
    now = datetime.now()
    current_time = now.strftime("%H%M%S")
    torch.save(cvae.state_dict(), "models/model" + current_time + 'z' + str(num_z))
    print("Saved model weights as model" + current_time + 'z' + str(num_z))
    return


def plotting(cvae, dataloader, num_z, fig_size=(10, 10)):
    _, (data, labels) = next(enumerate(dataloader))
    labels2 = F.one_hot(labels, num_classes=10)
    cvae.eval()
    device = torch.device('cpu')
    xhat, z = cvae(data, labels2, device)
    z = z.detach().numpy()
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
        fig.suptitle('latent space z')
        fig.show()

    plt.figure(figsize=fig_size)
    for i in range(num_z):
        plt.subplot(num_z, 1, i+1)
        plt.hist(z[:, i])
    plt.tight_layout()
    plt.show()

    x_clean_example = data[0:10, :, :, :]
    output_example = xhat[0:10, :, :, :]
    output_example = output_example.detach().numpy()

    plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)
    plt.hist(x_clean_example[0, :, :, :].flatten())
    plt.subplot(2, 1, 2)
    plt.hist(output_example[0, :, :, :].flatten())
    plt.show()

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
    plt.suptitle('output of the auto-encoder(num_z='+str(num_z)+') with respect to input numbers' + str(labels[0:10].numpy()))
    plt.show()
    return


def plot_zspace(cvae, number=3, num_samples=64):
    points_per_dim = torch.pow(torch.tensor(num_samples), 1/3)
    x = torch.linspace(-0.5, 0.5, int(points_per_dim.item()))
    plt.figure(figsize=(12, 12))
    for i in range(num_samples):
        latent_vars = torch.tensor([x[int(np.floor(i / points_per_dim.item()**2) % points_per_dim.item()**2)],
                                    x[int(np.floor(i/points_per_dim.item()) % points_per_dim.item())],
                                    x[i % int(points_per_dim.item())]])

        label = torch.tensor([number])
        label = F.one_hot(label, num_classes=10).squeeze()

        out = cvae.decoder(torch.unsqueeze(torch.cat([latent_vars, label], dim=-1), 0))

        plt.subplot(int(np.sqrt(num_samples)), int(np.sqrt(num_samples)), i+1)
        plt.imshow(np.squeeze(out.detach().numpy()), cmap='gray')
        plt.xticks([])
        plt.yticks([])

    plt.tight_layout()
    plt.show()

    return

