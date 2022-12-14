import torch
from torchmetrics import MeanSquaredError
import matplotlib.pyplot as plt


class SA:
    def __init__(self, decoder,  device, args, burn_in_frac=0.2, mc_sigma=0.1):
        self.decoder = decoder.to(device)
        self.device = device
        self.num_z = args.num_z
        self.num_samples = args.mh_steps
        self.burn_in = int(burn_in_frac * args.mh_steps)
        self.mc_sigma = mc_sigma

    def trial(self, new_loss, curr_loss, exp_amp=35):
        return torch.rand((1, 1), device=self.device) < torch.exp(-exp_amp*((new_loss - curr_loss)/curr_loss)) or new_loss < curr_loss

    def calc_loss(self, latent_point, image, mask, error_fn):
        with torch.no_grad():
            image_generated = self.decoder(latent_point)
        image_generated = image_generated.squeeze()
        recon_loss = error_fn(image_generated[mask], image[mask])
        return recon_loss

    def generate_mc_samples(self, image, mask, class_onehot, error_fn):
        if torch.sum(mask).item():
            stored_latent_pts = torch.zeros((1, self.num_z), device=self.device)
            error = torch.tensor(float('Inf'), device=self.device) * torch.ones(1, device=self.device)
            while len(stored_latent_pts) < self.num_samples + self.burn_in:
                latent_point = stored_latent_pts[-1, :] + torch.normal(0, self.mc_sigma, size=(1, self.num_z), device=self.device)
                error_candidate = self.calc_loss(torch.cat([latent_point, torch.unsqueeze(class_onehot, 0)], dim=1), image, mask, error_fn)
                acceptance = self.trial(error_candidate, error[-1])
                if acceptance:
                    error = torch.cat([error, error_candidate * torch.ones(1, device=self.device)], dim=0)
                    stored_latent_pts = torch.cat([stored_latent_pts, latent_point])
        else:
            stored_latent_pts = torch.randn((self.num_samples + self.burn_in, self.num_z), device=self.device)
            error = torch.tensor(1e10) * torch.ones((self.num_samples + self.burn_in, 1), device=self.device)
        return stored_latent_pts[self.burn_in:, :], error[-int(self.num_samples*0.1):]

    def sample(self, image, mask, class_onehot, mse, num_tries=3):
        stored_err = torch.tensor(float('Inf'))
        for _ in range(num_tries):
            latent_pts, final_err = self.generate_mc_samples(image, mask, class_onehot, mse)
            if torch.mean(final_err) < stored_err:
                stored_latent_pts = latent_pts
                stored_err = torch.mean(final_err)
        return stored_latent_pts

    def find_entropy(self, image, latent_pts, class_onehot, error_fn):
        images_gen = self.decoder(torch.cat([latent_pts, class_onehot.repeat(len(latent_pts), 1)], -1))
        images_truth = image.unsqueeze(0).repeat(len(latent_pts), 1, 1)
        loss = error_fn(torch.squeeze(images_gen), images_truth)
        return loss

    def algorithm(self, image, label):
        image = image.squeeze()
        image = image.to(self.device)
        mask = torch.zeros(1, 2)
        mse = MeanSquaredError().to(self.device)
        classes = torch.nn.functional.one_hot(torch.arange(10))
        mask_elements = torch.unsqueeze(torch.arange(len(image)).repeat(len(image), 1).flatten(), -1)
        mask_elements = torch.cat([mask_elements, torch.unsqueeze(torch.repeat_interleave(torch.arange(len(image)), len(image)), -1)], -1)
        mask = mask_elements
        # mask = torch.zeros(1, 2)
        loss = torch.ones(len(classes)) * torch.tensor(float('Inf'))
        for i in range(len(classes)):
            print('Evaluating ' + str(i))
            curr_class = classes[i, :]
            latent_pts = self.sample(image, mask, curr_class, mse, num_tries=1)
            loss[i] = self.find_entropy(image, latent_pts, curr_class, mse)

        print(torch.nn.functional.softmax(loss, dim=0))
        print('Belief: ' + str(torch.argmin(loss).detach().numpy()) + ' Truth: ' + str(label.detach().numpy()))




