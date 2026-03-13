import os
import pathlib

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# input action에 noise를 줌
class SinusoidalPositionEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.dim = dim

    def forward(self, t):
        # diffusion에서는 dim을 half - 1로 대신함
        half = self.dim // 2
        emb = math.log(10000.0) / (half - 1)
        emb = torch.exp( torch.arange(half, device=t.device) * -emb)
        # None: 차원을 하나 늘려줌 = unsqueeze
        emb = t[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

# in_c: obs / out_c: action chunck
# action: (batch, action_chunk, action_dim)
class ObsvertaionBlock(nn.Module):
    def __init__(self, in_c, pos_dim, out_c,):
        super().__init__()

        self.ln1 = nn.Linear(in_c + pos_dim, 64)
        self.ln2 = nn.Linear(64, 64)

        self.a = nn.Linear(64, out_c)
        self.b = nn.Linear(64, out_c)

        self.silu = nn.SiLU()

    def forward(self, obs):
        x = self.silu(self.ln1(obs))
        x = self.silu(self.ln2(x))

        a = self.a(x)
        b = self.b(x)

        return a, b
    
# action: (batch, action_chunk, action_dim)
class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.conv1 = nn.Conv1d(in_c, out_c, 3, 1, 1)

    def forward(self, x_t):
        return self.conv1(x_t)
        
class DiffusionPolicy(nn.Module):
    def __init__(self, action_dim, obs_dim, obs_latent, pos_latent):
        super().__init__()

        self.conv_block1 = ConvBlock(action_dim, obs_latent)

        self.conv_block2 = ConvBlock(obs_latent, obs_latent)
        self.pos_emd = SinusoidalPositionEmbedding(pos_latent)
        self.obs_block1 = ObsvertaionBlock(obs_dim, pos_latent, obs_latent)

        self.conv_block3 = ConvBlock(obs_latent, obs_latent)

        self.conv_block4 = ConvBlock(obs_latent, obs_latent)
        self.obs_block2 = ObsvertaionBlock(obs_dim, pos_latent, obs_latent)

        self.conv_block5 = ConvBlock(obs_latent, action_dim)

        self.silu = nn.SiLU()

    @staticmethod
    def get_b_a_ab(start, end, timestep):
        betas = torch.linspace(start, end, timestep)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        return betas, alphas, alpha_bars

    @staticmethod
    def q_sample(x, t, alpha_bars):
        noise = torch.randn_like(x)
        x_t = torch.sqrt(alpha_bars[t][:, None, None, None]) * x 
        + torch.sqrt(1 - alpha_bars[t][:, None, None, None]) * noise
        return x_t
    
    # x_t: noise가 들어간 x
    def forward(self, x_t, t, obs):
        z = self.silu(self.conv_block1(x_t))
        z = self.conv_block2(z)
    
        emb = self.pos_emd(t)
        obs_t = torch.cat((obs, emb), dim=-1)

        a, b = self.obs_block1(obs_t)

        z_a = a.unsqueeze(-1) * z
        z_ab = z_a + b.unsqueeze(-1)

        z = self.silu(self.conv_block3(z_ab))
        z = self.conv_block4(z)

        a, b = self.obs_block2(obs_t)

        z_a = a.unsqueeze(-1) * z
        z_ab = z_a + b.unsqueeze(-1)

        return self.conv_block5(z_ab)
