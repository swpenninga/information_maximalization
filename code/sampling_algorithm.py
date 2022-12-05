import math

import torch
from torchmetrics import MeanSquaredError
import matplotlib.pyplot as plt


class SA:
    def __init__(self, decoder, device, args, num_samples=100, burn_in_frac=0.2, mc_sigma=0.1):
        self.decoder = decoder.to(device)
        self.device = device
        self.num_z = args.num_z
        self.num_samples = num_samples
        self.burn_in = int(burn_in_frac * num_samples)
        self.mc_sigma = mc_sigma

    def trial(self, new_loss, curr_loss):
        return max(torch.rand((1, 1), device=self.device) < torch.exp(new_loss - curr_loss),
                   new_loss > curr_loss)

    def calc_loss(self, t, latent_point, image, mask, error_fn, constant):
        with torch.no_grad():
            image_generated = self.decoder(latent_point)
        image_generated = image_generated.squeeze()
        values_gen = image_generated[mask.nonzero(as_tuple=True)]
        values_real = image[mask.nonzero(as_tuple=True)]

        recon_loss = error_fn(values_gen, values_real)/(2*self.mc_sigma)
        const_loss = t*constant
        prior_loss = torch.sum(torch.square(latent_point), 1) / 20
        return - const_loss - recon_loss - prior_loss

    def generate_mc_samples(self, image, mask, error_fn):
        if torch.sum(mask).item():
            stored_latent_pts = torch.zeros((1, 2*self.num_z), device=self.device)
            error = torch.tensor(float('-Inf'))
            constant = 0.5 * (self.num_z+1 * math.log(2*math.pi) + math.log(self.mc_sigma))
            while len(stored_latent_pts) < self.num_samples + self.burn_in:
                latent_point = torch.cat([torch.randn((1, self.num_z), device=self.device),
                                          stored_latent_pts[-1, self.num_z:] +
                                          torch.normal(0, self.mc_sigma, size=(1, self.num_z), device=self.device)]
                                         , dim=1)
                error_candidate = self.calc_loss(int(torch.sum(mask)), latent_point, image, mask, error_fn, constant)
                acceptance = self.trial(error_candidate, error)
                if acceptance:
                    error = error_candidate
                    stored_latent_pts = torch.cat([stored_latent_pts, latent_point], dim=0)
        else:
            stored_latent_pts = torch.randn((self.num_samples + self.burn_in, 2*self.num_z), device=self.device)

        return stored_latent_pts[self.burn_in:, :], self.decoder(stored_latent_pts[self.burn_in:, :])

    def plt_latent(self, latent_pts):
        latent_pts = latent_pts.cpu().detach().numpy()
        fig = plt.figure(figsize=[10, 10])
        ax = fig.add_subplot(projection='3d')
        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
        ax.set_zlim(-3, 3)
        scatter = ax.scatter(latent_pts[:, 3], latent_pts[:, 4], latent_pts[:, 5])
        fig.suptitle('positions of samples')
        fig.show()

    def sample(self, image):
        mask = torch.zeros_like(image, device=self.device)
        mask[:, :] = 1
        mse = MeanSquaredError().to(self.device)
        plt.imshow(image[:, :].cpu().detach().numpy().squeeze())
        plt.show()
        for _ in range(10):
            latent_pts, generated_imgs = self.generate_mc_samples(image, mask, mse)
            self.plt_latent(latent_pts)
            plt.imshow(generated_imgs[-1, :, :, :].cpu().detach().numpy().squeeze())
            plt.show()
