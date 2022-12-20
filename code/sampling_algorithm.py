import torch
from torchmetrics import MeanSquaredError
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import multiprocessing as mp


class SA:
    def __init__(self, decoder,  device, args, burn_in_frac=0, mc_sigma=0.1, num_img_frac=0.2):
        self.decoder = decoder.to(device)
        self.device = device
        self.num_z = args.num_z
        self.num_samples = args.mh_steps
        self.burn_in = int(burn_in_frac * args.mh_steps)
        self.mc_sigma = mc_sigma
        self.num_img_frac = num_img_frac

    def trial(self, new_loss, curr_loss, exp_amp=15):
        return torch.rand((1, 1), device=self.device) < torch.exp(-exp_amp*((new_loss - curr_loss)/curr_loss)) or new_loss < curr_loss

    def calc_loss(self, latent_point, image, mask, error_fn):
        with torch.no_grad():
            image_generated = self.decoder(latent_point)
        image_generated = image_generated.squeeze()
        recon_loss = 0
        for i in range(len(mask)):
            recon_loss += error_fn(image_generated[mask[i, :]], image[mask[i, :]])
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
        print('Current loss: ')
        print(str(loss.cpu().detach().numpy()))
        print('Belief: ' + str(torch.argmin(loss[:, 1]).cpu().detach().numpy()) + ' Truth: ' + str(
            label.cpu().detach().numpy()))
        return

    def evaluate_classes(self, image, mask, classes, loss_fn):
        loss = torch.empty([len(classes), 2], device=self.device)
        loss[:, 0] = torch.arange(len(classes), device=self.device)
        loss[:, 1] = torch.ones(len(classes), device=self.device) * torch.tensor(float('Inf'), device=self.device)
        num_img = int(self.num_samples * self.num_img_frac)
        images_all_classes = torch.empty([len(classes), num_img, 1, len(image), len(image)], device=self.device)

        for i in range(len(classes)):
            curr_class = classes[i, :]
            latent_pts = self.sample(image, mask, curr_class, loss_fn, num_tries=1)
            loss[i, 1] = self.find_entropy(image, latent_pts, curr_class, loss_fn)

            with torch.no_grad():
                images_gen = self.decoder(torch.cat([latent_pts[-num_img:, :], curr_class.repeat(num_img, 1)], -1))
            images_all_classes[i, :, :, :, :] = images_gen
        loss[:, 1] = loss[:, 1] / torch.sum(loss[:, 1])

        return loss, images_all_classes

    def unseen_pixels(self, mask, image):
        mask_elements = torch.unsqueeze(torch.arange(len(image)).repeat(len(image), 1).flatten(), -1)
        mask_elements = torch.cat(
            [mask_elements, torch.unsqueeze(torch.repeat_interleave(torch.arange(len(image)), len(image)), -1)], -1)
        mask_elements = mask_elements.to(self.device)
        pixels_in_common = torch.where((mask_elements == mask).all(dim=1))[0]
        mask_elements = torch.cat((mask_elements[:pixels_in_common], mask_elements[pixels_in_common + 1:]))
        return mask_elements

    def algorithm(self, image, label, loss_fn='l1'):
        self.decoder.eval()
        image_real = image.squeeze()
        image_real = image_real.to(self.device)
        label = label.to(self.device)
        classes = torch.nn.functional.one_hot(torch.arange(10, device=self.device))
        mask = torch.zeros((1, 2), device=self.device)
        if loss_fn == 'mse':
            loss_fn = MeanSquaredError().to(self.device)
        elif loss_fn == 'l1':
            loss_fn = torch.nn.L1Loss().to(self.device)
        else:
            raise ValueError

        # Current status:
        loss, images_all_classes = self.evaluate_classes(image_real, mask.long(), classes, loss_fn)
        images_all_classes = torch.reshape(images_all_classes, [len(classes)*int(self.num_samples * self.num_img_frac), len(image_real), len(image_real)])
        self.print_status(mask, loss, label)

        # Find next move:
        mask_candidates = self.unseen_pixels(mask, image_real)
        candidate_loss = torch.zeros([len(mask_candidates), len(classes)], device=self.device)
        torch.save(mask_candidates, 'data/mask_candidates.pt')
        for idx, candidate in enumerate(mask_candidates):
            mask_candidate = torch.cat([mask, torch.unsqueeze(candidate, 0)], dim=0)

            now = datetime.now()
            current_time = now.strftime("%H:%M:%S")
            print('Testing mask candidate: ' + str(idx+1) + '/' + str(len(mask_candidates)) + '| Current time: ' + str(current_time))

            for i in range(len(images_all_classes)):
                loss, _ = self.evaluate_classes(images_all_classes[i, :, :], mask_candidate.long(), classes, loss_fn)
                candidate_loss[idx, :] += loss[:, 1]
        torch.save(candidate_loss, 'data/candidate_loss.pt')

# Test a model that was trained with L1 loss z16







