import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

class PositionEmbedding(nn.Module):
    def __init__(self, dim, scale):
        super().__init__()

        self.dim = dim
        self.scale = scale

        self.register_buffer(
            "freq",
            torch.randn(dim // 2) * scale
        )

    def forward(self, t):
        angles = t[:, None] + self.freqs[None]
        emb = torch.cat((torch.sin(angles), torch.cos(angles)), dim=1)
        return emb

class ResidualBlock(nn.Module):
    def __init__(self, in_c, out_c, t_emb_dim, dropout=0.1):
        super().__init__()

        assert in_c % 32 == 0, "don't insert in_c (in_c can div 32)"
        assert out_c % 32 == 0, "don't insert in_c (out_c can div 32)"

        self.norm1 = nn.GroupNorm(32, in_c) # 
        self.conv1 = nn.Conv2d(in_c, out_c, 3, padding=1)

        self.norm2 = nn.GroupNorm(32, out_c)
        self.dropout1 = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, padding=1)

        self.t_linear = nn.Linear(t_emb_dim, out_c)
        self.skip = nn.Conv2d(in_c, out_c, 1) if in_c != out_c else nn.Identity()

        self.silu = nn.SiLU()

    def forward(self, x, t):
        h = self.conv1(self.silu(self.norm1(x)))
        h = h + self.t_linear(t)[:, :, None, None]
        h = self.conv2(self.dropout1(self.silu(self.norm2(h))))
        return h + self.skip(x)

class AttentionBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        assert ch % 32 == 0, ""

        self.norm1 = nn.GroupNorm(32, ch)
        self.qkv = nn.Conv2d(ch, ch*3, 1)

        self.output = nn.Conv2d(ch, ch, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm1(x)
        qkv = self.qkv(h)
        q, k, v = qkv.chunk(3, dim=1)

        q = q.reshape(B, C, H * W)
        k = k.reshape(B, C, H * W)
        v = v.reshape(B, C, H * W)

        attn = torch.matmul(q.transpose(1, 2), k)
        v_t = v.transpose(1, 2)

        out_t = torch.matmul(attn, v_t)
        out = out_t.transpose(1, 2)    
        out = out.reshape(B, C, H, W)
        return out


class UpNet(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)

class DownNet(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(
        self, 
        in_c,
        base_c,
        c_mult,
        attn_res,
        t_emb,
    ):
        super().__init__()