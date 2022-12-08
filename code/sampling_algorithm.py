import torch
from torchmetrics import MeanSquaredError
import matplotlib.pyplot as plt


class SA_conditional:
    def __init__(self, decoder, classifier, device, args, burn_in_frac=0.2, mc_sigma=0.2):
        self.decoder = decoder.to(device)
        self.classifier = classifier.to(device)
        self.device = device
        self.num_z = args.num_z
        self.num_samples = args.mh_steps
        self.burn_in = int(burn_in_frac * args.mh_steps)
        self.mc_sigma = mc_sigma


class SA_classifier:
    def __init__(self, decoder, classifier, device, args, burn_in_frac=0.2, mc_sigma=0.2):
        self.decoder = decoder.to(device)
        self.classifier = classifier.to(device)
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
            error = torch.tensor(1e10) * torch.ones((self.num_samples + self.burn_in, 1), device=self.device)
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
        stored_err = torch.tensor(float('Inf'))
        for _ in range(num_tries):
            latent_pts, generated_imgs, final_err = self.generate_mc_samples(image, mask, mse)
            if torch.mean(final_err) < stored_err:
                stored_latent_pts = latent_pts
                stored_generated_imgs = generated_imgs
                stored_err = torch.mean(final_err)
        return stored_latent_pts, stored_generated_imgs

    def class_information(self, images):
        with torch.no_grad():
            prediction = self.classifier(images)
        value, _ = torch.mode(torch.argmax(prediction, 1))
        certainty = len(torch.argmax(prediction, 1)[torch.argmax(prediction, 1) == value])/len(prediction)*100
        return torch.mean(torch.sum(prediction, 1)), value, certainty

    def sampling_algorithm(self, dataloader, precision=0.55):
        idx, (image, _, label) = next(enumerate(dataloader))
        image = image.squeeze()
        image = image.to(self.device)
        mask = torch.zeros_like(image)

        certainty = 0
        while certainty < precision:
            latent_pts, gen_images = self.sample(image, mask, num_tries=15)
            _, value, certainty = self.class_information(gen_images)
            print(f'True Value: {label.cpu().detach().numpy()}')
            print(f'Current prediction: [{value}]')
            print(f'Certainty: [{certainty:.2f}%]')
            print(f'Number of pixels used: [{int(torch.sum(mask[:]))}]')
            max_information = 0
            for j in (mask == 0).nonzero():
                print(j.cpu().detach().numpy())
                proposal_mask = mask
                proposal_mask[j] = 1
                current_information = 0
                for i in range(int(len(gen_images)/10)):
                    latent_pts_gen2, gen_images_gen2 = self.sample(torch.squeeze(gen_images[-i, :, :, :]), proposal_mask, num_tries=1)
                    info_in_gen2, _, _ = self.class_information(gen_images_gen2)
                    current_information += torch.abs(info_in_gen2)
                if current_information > max_information:
                    max_information = current_information
                    new_mask = proposal_mask
            mask = new_mask
