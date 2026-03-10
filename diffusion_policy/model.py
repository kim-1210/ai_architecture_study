import os
import pathlib

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

class SinusoidalPositionEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.dim = dim

    def forward(self, t):
        # diffusion에서는 dim을 half - 1로 대신함
        half = self.dim // 2
        emb = math.log(10000.0) / (half - 1)
        emb = torch.exp( torch.arange(half) * -emb)
        # None: 차원을 하나 늘려줌 = unsqueeze
        emb = t[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb
    
class ResidualBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()



class DiffusionPolicy(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()