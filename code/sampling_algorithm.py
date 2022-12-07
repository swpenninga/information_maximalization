import math

import torch
from torchmetrics import MeanSquaredError
import matplotlib.pyplot as plt


class SA:
    def __init__(self, decoder, device, args, burn_in_frac=0.5, mc_sigma=0.2):
        self.decoder = decoder.to(device)
        self.device = device
        self.num_z = args.num_z
        self.num_samples = args.mh_steps
        self.burn_in = int(burn_in_frac * args.mh_steps)
        self.mc_sigma = mc_sigma

    def trial(self, new_loss, curr_loss, exp_amp=100):
        return torch.rand((1, 1), device=self.device) < torch.exp(-exp_amp*(new_loss - curr_loss)) or new_loss < curr_loss

    def calc_loss(self, latent_point, image, mask, error_fn):
        with torch.no_grad():
            image_generated = self.decoder(latent_point)
        image_generated = image_generated.squeeze()
        values_gen = image_generated[mask.nonzero(as_tuple=True)]
        values_real = image[mask.nonzero(as_tuple=True)]
        recon_loss = error_fn(values_gen, values_real)/(2*self.mc_sigma)
        return recon_loss

    def generate_mc_samples(self, image, mask, error_fn):
        if torch.sum(mask).item():
            stored_latent_pts = torch.zeros((1, self.num_z), device=self.device)
            error = torch.tensor(float('Inf'), device=self.device) * torch.ones(1, device=self.device)
            while len(stored_latent_pts) < self.num_samples + self.burn_in:
                latent_point = stored_latent_pts[-1, :] + torch.normal(0, self.mc_sigma, size=(1, self.num_z), device=self.device)
                error_candidate = self.calc_loss(latent_point, image, mask, error_fn)
                acceptance = self.trial(error_candidate, error[-1])
                if acceptance:
                    error = torch.cat([error, error_candidate * torch.ones(1, device=self.device)], dim=0)
                    stored_latent_pts = torch.cat([stored_latent_pts, latent_point])
        else:
            stored_latent_pts = torch.randn((self.num_samples + self.burn_in, self.num_z), device=self.device)
            error = torch.tensor(1000) * torch.ones((self.num_samples + self.burn_in, 1), device=self.device)
        return stored_latent_pts[self.burn_in:, :], self.decoder(stored_latent_pts[self.burn_in:, :]), error[-int(self.num_samples*0.1):]

    def plt_latent(self, latent_pts):
        latent_pts = latent_pts.cpu().detach().numpy()
        fig = plt.figure(figsize=[10, 10])
        ax = fig.add_subplot(projection='3d')
        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
        ax.set_zlim(-3, 3)
        scatter = ax.scatter(latent_pts[:, 0], latent_pts[:, 1], latent_pts[:, 2])
        fig.suptitle('positions of samples')
        fig.show()

    def sample(self, image, mask, num_tries=5):
        mse = MeanSquaredError().to(self.device)

    ####
        plt.imshow(image[:, :].cpu().detach().numpy().squeeze())
        plt.show()
        plt.hist(image[:, :].cpu().detach().numpy().squeeze())
        plt.show()
    ####

        stored_err = torch.tensor(float('Inf'))
        for _ in range(num_tries):
            latent_pts, generated_imgs, final_err = self.generate_mc_samples(image, mask, mse)
            if torch.mean(final_err) < stored_err:
                stored_latent_pts = latent_pts
                stored_generated_imgs = generated_imgs
                stored_err = torch.mean(final_err)

    ####
            self.plt_latent(latent_pts)
            plt.imshow(generated_imgs[-1, :, :, :].cpu().detach().numpy().squeeze())
            plt.show()
            plt.hist(generated_imgs[-1, :, :, :].cpu().detach().numpy().squeeze())
            plt.show()
    ####
        return stored_latent_pts, stored_generated_imgs