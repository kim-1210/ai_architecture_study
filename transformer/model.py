import os

import numpy as np
import math
import scipy

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.transforms as transformers

class InputEmbeddings(nn.Module):
    def __init__(self, dim, vocab_size):
        super().__init__()

        self.dim = dim
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, dim)

    def forward(self, x):
        # sqrt: dim이 큰 모델에서 임베딩이 너무 작은 신호가 되지 않도록 sqrt로 키워주는 것
        return self.embedding(x) * math.sqrt(self.dim) # 

class PositionEmbedding(nn.Module):
    def __init__(self, dim, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, dim, 2) * (-math.log(10000.0)/dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)

        self.register_buffer(pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class LayerNormalization(nn.Module):
    def __init__(self, dim, eps):
        super().__init__()

        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim= True)
        std = x.std(dim=-1, keepdim= True)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias
    
class FeedForwardBlock(nn.Module):
    def __init__(self, dim, d_ff, dropout):
        super().__init__()
        self.linear1 = nn.Linear(dim, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, dim)

    def forward(self, x):
        return self.linear2(
            self.dropout(
                torch.relu(
                    self.linear1(x)
                )
            )
        )

class MultiHeadAttention(nn.Module):
    def __init__(self, dim, n_head, dropout):
        super().__init__()

        self.dim = dim
        self.h = n_head

        assert dim % n_head == 0, "d_model is not divisible by h"

        self.d_k = dim // n_head
        self.w_q = nn.Linear(dim, dim, bias=False)
        self.w_k = nn.Linear(dim, dim, bias=False)
        self.w_v = nn.Linear(dim, dim, bias=False)
        self.w_o = nn.Linear(dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, mask, dropout):
        d_k = query.shape[-1]
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k) # query와 key의 유사도
        if mask is not None:
            attention_scores.masked_fill_(mask==0, -1e9)
        attention_scores = attention_scores.softmax(dim=-1)
        if dropout is not None:
            attention_scores = dropout(attention_scores)
        return (attention_scores @ value), attention_scores
    
    def forward(self, q, k, v, mask):
        query = self.w_q(q)
        key = self.w_k(k)
        value = self.w_v(v)

        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)

        x, self.attention_score = MultiHeadAttention.attention(query, key, value, mask, self.dropout)

        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)
        return self.w_o(x)

class ResidualConnection(nn.Module):
    def __init__(self, dim, dropout):
        super().__init__()

        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization(dim)

    def forward(self, x, sublayer):
        return x + self.dropout( sublayer( self.norm(x) ) )

class EncoderBlock(nn.Module):
    def __init__(self, dim, self_attention_block, feed_forward_block, dropout):
        super().__init__()

        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        # regidual block을 2개 생성
        self.residual_connections = nn.ModuleList(
            [ResidualConnection(dim, dropout) for _ in range(2)]
        )

    def forward(self, x, src_mask):
        x = self.residual_connections[0](
            x, lambda x: self.self_attention_block(x, x, x, src_mask)
        )
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x

class Encoder(nn.Module):
    def __init__(self, dim, layers):
        super().__init__()

        self.layers = layers
        self.norm = LayerNormalization(dim)

    def froward(self, x, src_mask):
        for layer in self.layers:
            x = layer(x, src_mask)
        return self.norm(x)

class DecoderBlock(nn.Module):
    def __init__(
        self, dim, self_attention_block, 
        cross_attention_block, feed_forward_block, dropout
    ):
        super().__init__()

        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block

        self.residual_connections = nn.ModuleList(
            [ResidualConnection(dim, dropout) for _ in range(3)]
        )

class Decoder(nn.Module):
    def __init__(self, dim, layers):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(dim)
    
    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)

class ProjectionLayer(nn.Module):
    def __init__(self, d_model, vocab_size) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x) -> None:
        return self.proj(x)
    
class Transformer(nn.Module):
    def __init__(self):
        super().__init__()