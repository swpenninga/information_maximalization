import torch
from torchmetrics import MeanSquaredError
import matplotlib.pyplot as plt


class SA:
    def __init__(self, decoder,  device, args):
        self.decoder = decoder.to(device)
        self.device = device
        self.num_z = args.num_z
        self.num_samples = args.mh_steps
        self.mc_sigma = args.mc_sigma
        self.num_pixels = args.num_pixels


    def mask_imgs(self, images, mask_set, len_classes, len_images):
        # This should be tested for masks longer than 1
        full_mask_set = mask_set.repeat(len_classes*len_images, 1, 1).long()
        pixels = images[full_mask_set]
        print(images.size())
        print(full_mask_set.size())
        print(images[0, 0, 0])
        print(images[0, 0, 1])
        return

    def gen_sample_set(self, image_set, mask_set, classes):
        # Format: (#parallel_chains, #dimensions + class, # current timestep)
        stored_latent_pts = torch.zeros((len(image_set)*len(mask_set)*len(classes), self.num_z + len(classes), 1), device=self.device)
        stored_latent_pts[:, self.num_z:, 0] = torch.Tensor.repeat(classes, int(len(stored_latent_pts)/len(classes)), 1)

        curr_latent_pts = stored_latent_pts[:, :, -1]
        curr_images = self.decoder(curr_latent_pts)
        self.mask_imgs(curr_images.squeeze(), mask_set, len(classes), len(image_set))


        cand_latent_pts = curr_latent_pts
        cand_latent_pts[:, :self.num_z] = curr_latent_pts[:, :self.num_z] + torch.normal(0, self.mc_sigma, size=(len(stored_latent_pts), self.num_z), device=self.device)
        images_out = self.decoder(cand_latent_pts)


        return stored_latent_pts

    def algorithm(self, image_real, label):
        self.decoder.eval()
        image_real = image_real.to(self.device)
        label = label.to(self.device)
        classes = torch.nn.functional.one_hot(torch.arange(10, device=self.device))
        mask = torch.zeros((1, 2, 2), device=self.device)
        mask[0, 1, 1] = 1
        print(mask)
        loss_fn = MeanSquaredError().to(self.device)

        # Create set of images based on current belief
        self.gen_sample_set(image_real, mask, classes)



        # Create set of images on belief of images (gen2)



        return