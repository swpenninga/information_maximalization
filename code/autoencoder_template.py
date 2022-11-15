import torch
import torch.nn as nn
import torch.nn.functional as F
# %%  Encoder

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 16, 3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(16, 16, 3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(16, 1, 3, stride=1, padding=1)

    def forward(self, x):
        # use the created layers here
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv3(x)), (2, 2))
        x = F.max_pool2d(self.conv4(x), (2, 4))
        return x


# %%  Decoder
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.conv5 = nn.ConvTranspose2d(1, 16, 3, stride=1, padding=1)
        self.conv6 = nn.ConvTranspose2d(16, 16, 3, stride=1, padding=1)
        self.conv7 = nn.ConvTranspose2d(16, 16, 3, stride=1, padding=1)
        self.conv8 = nn.ConvTranspose2d(16, 1, 3, stride=1, padding=1)
        self.upsample1 = nn.Upsample(scale_factor=(2, 4), mode='nearest')
        self.upsample2 = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x):
        x = self.upsample1(F.relu(self.conv5(x)))
        x = self.upsample2(F.relu(self.conv6(x)))
        x = self.upsample2(F.relu(self.conv7(x)))
        x = self.upsample2(self.conv8(x))
        return x


# %%  Autoencoder
class AE(nn.Module):
    def __init__(self):
        super(AE, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        h = self.encoder(x)
        r = self.decoder(h)
        return r, h


