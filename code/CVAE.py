import torch
import torch.nn as nn
import torch.nn.functional as F


# %%  Encoder
class Encoder(nn.Module):
    def __init__(self, num_z):
        super(Encoder, self).__init__()
        self.filters = 64
        self.conv1 = nn.Conv2d(1, self.filters, 4, stride=1, padding=1)
        self.conv2 = nn.Conv2d(self.filters, self.filters, 4, stride=1, padding=1)
        self.conv3 = nn.Conv2d(self.filters, self.filters, 4, stride=1, padding=1)
        self.fc1 = nn.Linear(self.filters*4, self.filters)
        self.fc2 = nn.Linear(self.filters+10, int(2*num_z))

    def forward(self, x, l):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))  # [N,1,13,13]
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))  # [N,filters,6,6]
        x = F.max_pool2d(F.relu(self.conv3(x)), (2, 2))  # [N,filters,2,2]
        x = F.relu(self.fc1(torch.flatten(x, start_dim=1)))
        x = self.fc2(torch.cat([x, l], dim=-1))
        return x


# %%  Decoder
class CDecoder(nn.Module):
    def __init__(self, num_z):
        super(CDecoder, self).__init__()
        self.filters = 64
        self.fc1 = nn.Linear(num_z + 10, self.filters*2)
        self.fc2 = nn.Linear(self.filters*2, self.filters*4)
        self.conv1 = nn.ConvTranspose2d(self.filters, self.filters, 4, stride=1, padding=1)
        self.conv2 = nn.ConvTranspose2d(self.filters, self.filters, 4, stride=1, padding=1)
        self.conv3 = nn.ConvTranspose2d(self.filters, self.filters, 3, stride=1, padding=1)
        self.conv4 = nn.ConvTranspose2d(self.filters, 1, 3, stride=1, padding=1)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')


    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.upsample(F.relu(self.conv1(torch.unflatten(x, 1, [self.filters, 2, 2]))))  # [N,filters,6,6]
        x = self.upsample(F.relu(self.conv2(x)))  # [N,filters,6,6]
        x = self.upsample(F.relu(self.conv3(x)))  # [N,filters,28,28]
        x = self.conv4(x)  # [N,1,28,28]
        return x


# %%  Autoencoder
class CVAE(nn.Module):
    def __init__(self, num_z):
        super(CVAE, self).__init__()
        self.encoder = Encoder(num_z)
        self.decoder = CDecoder(num_z)

    def forward(self, x, l, device):
        x = self.encoder(x, l)
        z = sample(x, device)
        xhat = self.decoder(torch.cat([z, l], dim=-1))
        return xhat, z


# %% Sample function for AEs
def sample(h, device):
    z = torch.empty(h.size()[0], int(h.size()[1]/2), device=device)
    for i in range(int(h.size()[1]/2)):
        z[:, i] = h[:, i] + torch.mul(torch.randn(h.shape[0], device=device), torch.exp(h[:, int(i+h.size()[1]/2)]))
    return z
