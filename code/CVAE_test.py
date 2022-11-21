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
        self.fc2 = nn.Linear(32, 12)
        self.fc3 = nn.Linear(12, int(2 * num_z))
        self.fc4 = nn.Linear(13, int(2 * num_z))
        self.bn = nn.BatchNorm2d(16)

    def forward(self, x, l):
        x = F.max_pool2d(F.relu(self.bn(self.conv1(x))), (2, 2))
        x = F.max_pool2d(F.relu(self.bn(self.conv2(x))), (2, 2))
        x = F.max_pool2d(F.relu(self.bn(self.conv3(x))), (2, 2))
        x = F.relu(self.fc1(torch.flatten(x, start_dim=1)))
        x = F.relu(self.fc2(x))
        x_z = self.fc3(x)
        x_c = self.fc4(torch.cat([x, torch.unsqueeze(l, -1)], -1))

        return x_z, x_c


# %%  Decoder
class Decoder(nn.Module):
    def __init__(self,num_z):
        super(Decoder, self).__init__()
        self.conv1 = nn.ConvTranspose2d(16, 16, 3, stride=1, padding=1)
        self.conv2 = nn.ConvTranspose2d(16, 16, 3, stride=1, padding=1)
        self.conv3 = nn.ConvTranspose2d(16, 1, 3, stride=1, padding=1)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.fc1 = nn.Linear(2*num_z, 32)
        self.fc2 = nn.Linear(32, 256)
        self.bn = nn.BatchNorm2d(16)

    def forward(self, x, c):
        x = F.relu(self.fc1(torch.cat([x, c], -1)))
        x = F.relu(self.fc2(x))
        x = self.upsample(F.relu(self.bn(self.conv1(x.reshape([int(x.size()[0]), 16, 4, 4])))))
        x = self.upsample(F.relu(self.bn(self.conv2(x))))
        x = self.upsample(self.conv3(x))
        return x


# %%  Autoencoder
class AE(nn.Module):
    def __init__(self,num_z):
        super(AE, self).__init__()
        self.encoder = Encoder(num_z)
        self.decoder = Decoder(num_z)

    def forward(self, x, l):
        x_z, x_c = self.encoder(x, l)
        z = sample(x_z)
        c = sample(x_c)
        xhat = self.decoder(z, c)
        return xhat, x_z, x_c, z, c


def sample(h):
    z = torch.empty(h.size()[0], int(h.size()[1]/2))
    for i in range(int(h.size()[1]/2)):
        z[:, i] = h[:, i] + torch.mul(torch.randn(h.shape[0]), torch.exp(h[:, int(i+h.size()[1]/2)]))
    return z
