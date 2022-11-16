import torch
import torch.nn as nn
import torch.nn.functional as F
# %%  Encoder


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 16, 3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(16, 1, 3, stride=1, padding=1)
        self.fc1 = nn.Linear(50, 1)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv3(x)), (2, 2))
        x = self.fc1(x)
        return x


# %%  Decoder
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.conv1 = nn.ConvTranspose2d(1, 16, 3, stride=1, padding=1)
        self.conv2 = nn.ConvTranspose2d(16, 16, 3, stride=1, padding=1)
        self.conv3 = nn.ConvTranspose2d(16, 1, 3, stride=1, padding=1)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.fc1 = nn.Linear(1, 50)

    def forward(self, x):
        x = self.fc1(x)
        x = self.upsample(F.relu(self.conv1(x)))
        x = self.upsample(F.relu(self.conv2(x)))
        x = self.upsample(F.relu(self.conv3(x)))
        return x


# %%  Autoencoder
class AE(nn.Module):
    def __init__(self):
        super(AE, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        z = self.encoder(x)
        r = self.decoder(z)
        return r, z


