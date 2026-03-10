import numpy as np
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()

        self.layers = [
            nn.Conv2d(input_dim, 64, 4, 2, 1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.Conv2d(64, 64, 4, 2, 1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
        ]

        self.backbone = nn.Sequential(*self.layers)
        self.to_mu = nn.Conv2d(64, latent_dim, 1, 1, 0)
        self.to_logvar = nn.Conv2d(64, latent_dim, 1, 1, 0)

    def forward(self, x):
        h = self.backbone(x)
        mu = self.to_mu(h)
        logvar = self.to_logvar(h)
        return mu, logvar
    
class Decoder(nn.Module):
    def __init__(self, latent_dim, output_dim, img_size):
        super().__init__()

        self.img_size = img_size

        self.layers = [
            nn.ConvTranspose2d(latent_dim, 64, 4, 2, 1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.ConvTranspose2d(64, 64, 4, 2, 1),
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

    @staticmethod
    def reparameterize(mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    @staticmethod
    def kl_loss(mu, logvar):
        return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    def encode_stats(self, x):
        return self.encoder(x)

    def encode(self, x):
        mu, logvar = self.encode_stats(x)
        return self.reparameterize(mu, logvar)
    
    def decode(self, z):
        return self.decoder(z)

    def forward(self, x, return_stats=False):
        mu, logvar = self.encode_stats(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        if return_stats:
            return x_recon, z, mu, logvar
        return x_recon, z
