import torch
from torchmetrics import MeanSquaredError
import matplotlib.pyplot as plt


class SA:
    def __init__(self, decoder, device, args, num_samples=100):
        self.decoder = decoder.to(device)
        self.device = device
        self.num_z = args.num_z
        self.num_samples = num_samples

    def generate_random_samples(self, image, mask, precision=0.01):
        mean_squared_error = MeanSquaredError().to(self.device)
        # Create placeholder for latent_space samples that pass the test
        stored_latent_pts = torch.empty((1, self.num_z), device=self.device)

        # Ignore first element, we want a num_samples amount of points
        while len(stored_latent_pts) < self.num_samples+1:
            latent_points = torch.randn(self.num_samples, self.num_z, device=self.device)
        # Generate image with randomized z parameters
            with torch.no_grad():
                images_generated = self.decoder(torch.cat([
                    torch.randn(self.num_samples, self.num_z, device=self.device), latent_points], -1))

        # Error placeholder for all the generated candidates
            error = torch.zeros(self.num_samples)
        # If there is an open pixel in the mask, mask both real and generated images
            if torch.sum(mask).item():
                for i in range(self.num_samples):
                    curr_image = images_generated[i].squeeze()
        # Calculate the MAPE between the pixels that are in the mask
                    values_gen = curr_image[mask.nonzero(as_tuple=True)]
                    values_real = image[mask.nonzero(as_tuple=True)]
                    error[i] = mean_squared_error(values_gen, values_real)
        # Record if the MAPE is below a threshold and store the complying latent_points
            passed_points = [j for j in range(len(error)) if error[j] < precision]
            stored_latent_pts = torch.cat([stored_latent_pts, latent_points[passed_points, :]], dim=0)

        stored_latent_pts = stored_latent_pts[1:self.num_samples+1, :]
        images_latent_pts = self.decoder(torch.cat([
            torch.randn(self.num_samples, self.num_z, device=self.device), stored_latent_pts], -1))
        return stored_latent_pts, images_latent_pts

    def naive_entropy(self, latent_img):
        latent_img = latent_img.squeeze()
        std_max = 0
        mask_pos = torch.tensor([12, 12])
        for i in range(latent_img.size()[1]):
            for j in range(latent_img.size()[2]):
                if torch.std(latent_img[:, i, j]) > std_max:
                    std_max = torch.std(latent_img[:, i, j])
                    mask_pos = torch.tensor([i, j])
        return mask_pos

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

    def sample(self, image):
        mask = torch.zeros_like(image, device=self.device)
        for i in range(15):
            latent_pts, latent_img = self.generate_random_samples(image, mask)
            self.plt_latent(latent_pts)
            next_measurement = self.naive_entropy(latent_img)
            mask[next_measurement[0], next_measurement[1]] = 1

