import torch
import math

class MH:
    def __init__(self, model, device, args):
        self.model = model
        self.device = device
        self.num_z = args.num_z

    def calc_loss(self, memory, sample):
        t = memory.numel()
        const_loss = -0.5*(t + self.num_z) * torch.log(torch.tensor(2*math.pi))
        prior_loss = -torch.sum(torch.square(sample))/2
        recons_loss = 0
        return const_loss + prior_loss + recons_loss

    def trans_step(self, sample, sigma_trans):
        return torch.normal(sample, torch.ones((1, self.num_z*2)) * sigma_trans)

    def trial(self, new_loss, curr_loss):
            return max(torch.rand((1, 1)) < torch.exp(new_loss - curr_loss),
                       new_loss > curr_loss)

    def sample(self, memory=None, init_sample=None, num_points=100, burn_ratio=0.2, sigma_trans=0.5):
        if memory is None:
            memory = []
        if init_sample is None:
            init_sample = torch.normal(0, 1, size=(1, self.num_z*2))
        burn_in = int(burn_ratio * num_points)
        curr_loss = self.calc_loss(memory, init_sample)

        sample = init_sample
        sample_set = sample
        log_posterior = curr_loss.unsqueeze(0)

        for _ in range(num_points):
            new_sample = self.trans_step(sample, sigma_trans)
            new_loss = self.calc_loss(memory, sample)
            accepted = self.trial(new_loss, curr_loss)
            if accepted:
                sample = new_sample
                curr_loss = new_loss
            sample_set = torch.cat((sample_set, sample))
            log_posterior = torch.cat((log_posterior, curr_loss.unsqueeze(0)))

        return sample_set[burn_in:], log_posterior[burn_in:]


def action_num(model, sample_set, memory, image_size, device):
    unique_samples, frequency = sample_set.unique(return_counts=True, dim=0)
    frequency = frequency/torch.sum(frequency)

    unique_samples = unique_samples.to(device)
    with torch.no_grad():
        sampled_images = model.decoder(unique_samples)
    possible_actions = torch.tensor([a for a in range(image_size) if a not in memory])
    possible_observations = torch.flatten(sampled_images, start_dim=2)[:, :, possible_actions]
    possible_observations = torch.t(torch.squeeze(possible_observations))
    taken_action = -torch.ones(image_size) * float('inf')
    for idx, action in enumerate(possible_actions):
        mu_a = possible_observations[idx]
        mu_k = mu_a.repeat(len(mu_a), 1)
        mu_l = mu_k.T

        kl_div = torch.exp(torch.square(mu_k-mu_l)/0.2)
        sum_inter = torch.sum(frequency*kl_div, 1)
        taken_action[action] = torch.sum(frequency*torch.log(sum_inter))
    return torch.argmax(taken_action, keepdim=True)




