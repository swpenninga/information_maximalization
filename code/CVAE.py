import torch
import torch.nn as nn
import torch.nn.functional as F
# %%  Encoder


class Encoder(nn.Module):
    def __init__(self,num_z):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 16, 3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(16, 16, 3, stride=1, padding=1)
        self.fc1 = nn.Linear(256, 32)
        self.fc2 = nn.Linear(32, int(2*num_z))

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv3(x)), (2, 2))
        x = F.relu(self.fc1(torch.flatten(x, start_dim=1)))
        x = self.fc2(x)
        return x


# %%  Decoder
class Decoder(nn.Module):
    def __init__(self,num_z):
        super(Decoder, self).__init__()
        self.conv1 = nn.ConvTranspose2d(16, 16, 3, stride=1, padding=1)
        self.conv2 = nn.ConvTranspose2d(16, 16, 3, stride=1, padding=1)
        self.conv3 = nn.ConvTranspose2d(16, 1, 3, stride=1, padding=1)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.fc1 = nn.Linear(num_z, 32)
        self.fc2 = nn.Linear(32, 256)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.upsample(F.relu(self.conv1(x.reshape([int(x.size()[0]), 16, 4, 4]))))
        x = self.upsample(F.relu(self.conv2(x)))
        x = self.upsample(self.conv3(x))
        return x


# %%  Autoencoder
class AE(nn.Module):
    def __init__(self,num_z):
        super(AE, self).__init__()
        self.encoder = Encoder(num_z)
        self.decoder = Decoder(num_z)

    def forward(self, x):
        h = self.encoder(x)
        z = sample(h)
        xhat = self.decoder(z)
        return xhat, h, z


def sample(h):
    z = torch.empty(h.size()[0], int(h.size()[1]/2))
    for i in range(int(h.size()[1]/2)):
        z[:, i] = h[:, i] + torch.mul(torch.randn(h.shape[0]), torch.exp(h[:, int(i+h.size()[1]/2)]))
    return z
