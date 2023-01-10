import functools

import torch
from torchmetrics import MeanSquaredError
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import concurrent.futures


class SA:
    def __init__(self, decoder,  device, args, burn_in_frac=0, num_img_frac=0.1, sigma_obs=0.1):
        self.decoder = decoder.to(device)
        self.device = device
        self.num_z = args.num_z
        self.num_samples = args.mh_steps
        self.burn_in = int(burn_in_frac * args.mh_steps)
        self.mc_sigma = args.mc_sigma
        self.num_img_frac = num_img_frac
        self.num_pixels = args.num_pixels
        self.sigma_obs = sigma_obs
        self.exp_amp = args.exp_amp

    def trial(self, new_loss, curr_loss):
        return torch.rand((1, 1), device=self.device) < torch.exp(-self.exp_amp*((new_loss - curr_loss)/curr_loss)) or new_loss < curr_loss

    def calc_loss(self, latent_point, image, mask, error_fn):
        with torch.no_grad():
            image_generated = self.decoder(latent_point)
        image_generated = image_generated.squeeze()
        recon_loss = 0
        for i in range(len(mask)):
            recon_loss += error_fn(image_generated[mask[i, :]], image[mask[i, :]])
        latent_pt_loss = torch.sum(torch.abs(latent_point))*1e-3
        return recon_loss + latent_pt_loss

    def generate_mc_samples(self, image, mask, class_onehot, error_fn):
        if torch.sum(mask).item():
            stored_latent_pts = torch.zeros((1, self.num_z), device=self.device)
            error = torch.tensor(float('Inf'), device=self.device) * torch.ones(1, device=self.device)
            while len(stored_latent_pts) < self.num_samples + self.burn_in:
                latent_point = stored_latent_pts[-1, :] + torch.normal(0, torch.max(torch.tensor(0.005), self.mc_sigma * torch.tensor((self.num_samples + self.burn_in - len(stored_latent_pts))/(self.num_samples + self.burn_in))), size=(1, self.num_z), device=self.device)
                error_candidate = self.calc_loss(torch.cat([latent_point, torch.unsqueeze(class_onehot, 0)], dim=1), image, mask, error_fn)
                acceptance = self.trial(error_candidate, error[-1])
                if acceptance:
                    error = torch.cat([error, error_candidate * torch.ones(1, device=self.device)], dim=0)
                    stored_latent_pts = torch.cat([stored_latent_pts, latent_point])

        else:
            stored_latent_pts = torch.randn((self.num_samples + self.burn_in, self.num_z), device=self.device)
            error = torch.tensor(1e10, device=self.device) * torch.ones((self.num_samples + self.burn_in, 1), device=self.device)
        return stored_latent_pts[self.burn_in:, :], error[-int(self.num_samples*0.1):]

    def sample(self, image, mask, class_onehot, lossfn, num_tries=3):
        stored_err = torch.tensor(float('Inf'))
        for _ in range(num_tries):
            latent_pts, final_err = self.generate_mc_samples(image, mask, class_onehot, lossfn)
            if torch.mean(final_err) < stored_err:
                stored_latent_pts = latent_pts
                stored_err = torch.mean(final_err)
        return stored_latent_pts

    def find_entropy(self, image, latent_pts, class_onehot, error_fn):
        with torch.no_grad():
            images_gen = self.decoder(torch.cat([latent_pts, class_onehot.repeat(len(latent_pts), 1)], -1))
        images_truth = image.unsqueeze(0).repeat(len(latent_pts), 1, 1)
        loss = error_fn(torch.squeeze(images_gen), images_truth)
        return loss

    def print_status(self, mask, loss, label):
        print('Num pixels sampled: ' + str(len(mask) - 1))
        print(f"Knowledge: {(len(mask) - 1)/(28**2) * 100:.2f}%")
        print('Belief: ' + str(torch.argmin(loss[:, 1]).cpu().detach().numpy()) + ' Truth: ' + str(
            label.cpu().detach().numpy()))
        print('Current loss: ')
        print(str(loss.cpu().detach().numpy()))
        print('- - - - - - - - - - - - - - - - -')
        return

    def evaluate_classes(self, image, mask, classes, loss_fn, num_tries=1):
        loss = torch.empty([len(classes), 2], device=self.device)
        loss[:, 0] = torch.arange(len(classes), device=self.device)
        loss[:, 1] = torch.ones(len(classes), device=self.device) * torch.tensor(float('Inf'), device=self.device)
        num_img = int(self.num_samples * self.num_img_frac)
        images_all_classes = torch.empty([len(classes), num_img, 1, len(image), len(image)], device=self.device)

        for i in range(len(classes)):
            curr_class = classes[i, :]
            latent_pts = self.sample(image, mask, curr_class, loss_fn, num_tries=num_tries)
            loss[i, 1] = self.find_entropy(image, latent_pts, curr_class, loss_fn)

            with torch.no_grad():
                images_gen = self.decoder(torch.cat([latent_pts[-num_img:, :], curr_class.repeat(num_img, 1)], -1))
            images_all_classes[i, :, :, :, :] = images_gen

        return loss, images_all_classes

    def unseen_pixels(self, mask, image):
        mask_elements = torch.unsqueeze(torch.arange(len(image)).repeat(len(image), 1).flatten(), -1)
        mask_elements = torch.cat(
            [mask_elements, torch.unsqueeze(torch.repeat_interleave(torch.arange(len(image)), len(image)), -1)], -1)
        mask_elements = mask_elements.to(self.device)
        for i in range(len(mask)):
            curr_mask = mask[i, :]
            pixels_in_common = torch.where((mask_elements == curr_mask).all(dim=1))[0]
            mask_elements = torch.cat((mask_elements[:pixels_in_common], mask_elements[pixels_in_common + 1:]))
        return mask_elements

    def decide_pixel(self, image_set, mask_candidates):
        Q_a = -torch.ones(len(mask_candidates)) * float('inf')
        unique_samples, frequency = image_set.unique(return_counts=True, dim=0)
        for item_idx, item in enumerate(mask_candidates):
            mu_a = image_set[:, item[0], item[1]]
            mu_k = mu_a.repeat(len(mu_a), 1)
            mu_l = mu_k.T
            kl_div = torch.exp(torch.square(mu_k - mu_l) / (2 * self.sigma_obs))
            sum_inter = torch.sum(frequency * kl_div, 1)
            Q_a[item_idx] = torch.sum(frequency * torch.log(sum_inter))

        return mask_candidates[torch.argmax(Q_a)]

    def algorithm(self, image, label):
        self.decoder.eval()
        image_real = image.squeeze()
        image_real = image_real.to(self.device)
        label = label.to(self.device)
        classes = torch.nn.functional.one_hot(torch.arange(10, device=self.device))
        mask = torch.zeros((1, 2), device=self.device)
        loss_fn = MeanSquaredError().to(self.device)

        while len(mask) < self.num_pixels+1:
            # Current status:
            loss, images_all_classes = self.evaluate_classes(image_real, mask.long(), classes, loss_fn, num_tries=2)
            self.print_status(mask, loss, label)

            # Next status:
            height = images_all_classes.size()[3]
            width = images_all_classes.size()[4]
            images_all_classes = images_all_classes.reshape(
                [int(self.num_samples * self.num_img_frac * len(classes)), height, width])
            mask_candidates = self.unseen_pixels(mask, image_real)
            chosen_pixel = self.decide_pixel(images_all_classes, mask_candidates)
            mask = torch.cat([mask, torch.unsqueeze(chosen_pixel, 0)], dim=0)

        now = datetime.now()
        current_time = now.strftime("%H%M%S")
        torch.save(image_real, 'data/image'+str(current_time)+'.pt')
        torch.save(mask, 'data/mask'+str(current_time)+'.pt')










