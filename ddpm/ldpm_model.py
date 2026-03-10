import numpy as np
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()

        self.layers = [
            nn.Conv2d(input_dim, 64, 3, 1, 1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.Conv2d(64, latent_dim, 3, 1, 1),
        ]

        self.model = nn.Sequential(*self.layers)

    def forward(self, x):
        z = self.model(x)
        return z
    
class Decoder(nn.Module):
    def __init__(self, latent_dim, output_dim, img_size):
        super().__init__()

        self.img_size = img_size

        self.layers = [
            nn.Conv2d(latent_dim, 64, 3, 1, 1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.Conv2d(64, output_dim, 3, 1, 1),
        ]

        self.model = nn.Sequential(*self.layers)

    def forward(self, z):
        x = self.model(z)
        x = F.interpolate(x, size=(self.img_size, self.img_size), mode="bilinear")
        return x

class VAE(nn.Module):
    def __init__(self, input_dim=3, latent_dim=128, output_dim=3, img_size=64):
        super().__init__()

        self.encoder = Encoder(input_dim, latent_dim)
        self.decoder = Decoder(latent_dim, output_dim, img_size)

    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        z = self.encode(x)
        x_recon = self.decode(z)
        return x_recon, z