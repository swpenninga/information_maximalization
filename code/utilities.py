import matplotlib.pyplot as plt
from datetime import datetime
import torch


def save_model(cvae):
    now = datetime.now()
    current_time = now.strftime("%H%M%S")
    torch.save(cvae.state_dict(), "models/model" + current_time)
    print("Saved model weights as model" + current_time + ".")
    return


def plotting(cvae, data, labels, num_z, fig_size=(10, 10)):
    cvae.eval()
    xhat, x_z, x_c, z, c = cvae(data, labels)
    z = z.detach().numpy()
    c = c.detach().numpy()
    if num_z == 3:
        fig = plt.figure(figsize=fig_size)
        ax = fig.add_subplot(projection='3d')
        ax.scatter(z[:, 0], z[:, 1], z[:, 2], c=labels, cmap='tab10')
        plt.suptitle('latent space of z')
        fig.show()

        fig = plt.figure(figsize=fig_size)
        ax = fig.add_subplot(projection='3d')
        ax.scatter(c[:, 0], c[:, 1], c[:, 2], c=labels, cmap='tab10')
        plt.suptitle('latent space of c')
        fig.show()
    elif num_z == 2:
        fig = plt.figure(figsize=fig_size)
        ax = fig.add_subplot()
        ax.scatter(z[:, 0], z[:, 1], c=labels, cmap='tab10')
        plt.suptitle('latent space of z')
        fig.show()

        fig = plt.figure(figsize=fig_size)
        ax = fig.add_subplot()
        ax.scatter(c[:, 0], c[:, 1], c=labels, cmap='tab10')
        plt.suptitle('latent space of c')
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
    plt.suptitle('output of the auto-encoder with respect to input numbers' + str(labels[0:10].numpy()))
    plt.show()