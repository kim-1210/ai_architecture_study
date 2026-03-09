import os

import numpy as np
import math
import scipy

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.transforms as transformers

# src: 입력문장 / tgt: 출력문장

class InputEmbeddings(nn.Module):
    def __init__(self, dim, vocab_size):
        super().__init__()

        self.dim = dim
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, dim)

    def forward(self, x):
        # sqrt: dim이 큰 모델에서 임베딩이 너무 작은 신호가 되지 않도록 sqrt로 키워주는 것
        return self.embedding(x) * math.sqrt(self.dim)

class PositionEmbedding(nn.Module):
    def __init__(self, dim, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        # y=ln(a) -> a=e^y -> a=e^ln(a) -> 
        # if a=10000^x, 10000^x=e^xln(10000)
        div_term = torch.exp(
            -1 * (torch.arange(0, dim, 2)/dim) * math.log(10000)
        )
        # 0::2 -> 0배열차례부터 시작해서 2step씩
        pe[:, 0::2] = torch.sin(position * div_term)
        # 1::2 -> 1배열차례부터 시작해서 2step씩
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)

        self.register_buffer(pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

# nn.Normalization과 같음
class LayerNormalization(nn.Module):
    def __init__(self, dim, eps):
        super().__init__()

        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        nn.Normalization

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim= True)
        std = x.std(dim=-1, keepdim= True)
        # eps: std가 0일때 0으로 나누어지는 것을 방지해줌
        return self.alpha * ((x - mean) / (std + self.eps)) + self.bias

# 단순 MLP
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

# attention을 다중으로 만들어서 시간을 획기적으로 줄임
class MultiHeadAttention(nn.Module):
    def __init__(self, dim, n_head, dropout):
        super().__init__()

        self.dim = dim
        self.h = n_head

        assert dim % n_head == 0, "head의 수는 dim과 나눴을떄 나머지가 0 이어야함"

        self.d_k = dim // n_head
        self.w_q = nn.Linear(dim, dim, bias=False)
        self.w_k = nn.Linear(dim, dim, bias=False)
        self.w_v = nn.Linear(dim, dim, bias=False)
        self.w_o = nn.Linear(dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, mask, dropout):
        d_k = query.shape[-1]
        # math.sqrt(d_k): 값이 너무 커지는 것을 방지해줌
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k) # query와 key의 유사도
        if mask is not None:
            # 문장 길이가 다른거나 문장을 생성할때 attention에 계산이 되면 안됨
            # mask가 0인 위치들은 표함이 되면 안되니 -1e9로 나오기 힘든 낮은 값으로 masking함
            attention_scores.masked_fill_(mask==0, -1e9)
        attention_scores = attention_scores.softmax(dim=-1)
        if dropout is not None:
            attention_scores = dropout(attention_scores)
        # attention_scores: 유사성, (attention_scores @ value): 얼마나 받아들일지 
        return (attention_scores @ value), attention_scores
    
    def forward(self, q, k, v, mask):
        query = self.w_q(q)
        key = self.w_k(k)
        value = self.w_v(v)

        # (batch, seq, n_head, dim/n_head)
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)

        x, self.attention_score = MultiHeadAttention.attention(query, key, value, mask, self.dropout)

        # (batch, n_head, seq, d_k) -> (batch, seq, dim)
        # transpose(1, 2): shape에서 1과 2의 배열값들을 바꿔줌
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)
        # 위 변형에서는 n_head에 대한 각정보들을 합쳤을 뿐, 섞지 않았음, w_o에서 정보들의 혼합
        return self.w_o(x)

# 원본인 x를 더함으로써 원본의 정보를 잃지않음
class ResidualConnection(nn.Module):
    def __init__(self, dim, dropout):
        super().__init__()

        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization(dim)

    def forward(self, x, sublayer):
        return x + self.dropout( sublayer( self.norm(x) ) )

# layer: self_attention, feed_forward, residual, residual
class EncoderBlock(nn.Module):
    def __init__(self, dim, self_attention_block, feed_forward_block, dropout):
        super().__init__()

        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        # residual block을 2개 생성
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

# layer: self_attention, cross_attention, feed_forward, residual, residual
class DecoderBlock(nn.Module):
    def __init__(
        self, dim, self_attention_block, 
        cross_attention_block, feed_forward_block, dropout
    ):
        super().__init__()
        # tgt의 과거 만 보고 다음 단어를 예측
        self.masked_self_attention_block = self_attention_block
        # 예측한 단어를 encoder의 output과 비교하여 예측 수정
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block

        self.residual_connections = nn.ModuleList(
            [ResidualConnection(dim, dropout) for _ in range(3)]
        )
    
    # x: tgt dim / src_mask: encoder에서 padding값을 처리할 크기 / tgt_mask: 미래의 정보를 숨길 크기
    def forward(self ,x, encoder_output, src_mask, tgt_mask):
        x = self.residual_connections[0](x, lambda x: self.masked_self_attention_block(x, x, x, tgt_mask))
        x = self.residual_connections[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, src_mask))
        x = self.residual_connections[2](x, self.feed_forward_block)
        return x

class Decoder(nn.Module):
    def __init__(self, dim, layers):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(dim)
    
    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)

# latent 값들을 output값으로 변형
class ProjectionLayer(nn.Module):
    def __init__(self, d_model, vocab_size) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x) -> None:
        return self.proj(x)
    
class Transformer(nn.Module):
    def __init__(
        self, encoder, decoder, src_embed, 
        tgt_embed, src_pos, tgt_pos, projection_layer
    ):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer

    def encode(self, src, src_mask):
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)
    
    def decode(self, encoder_output, src_mask, tgt, tgt_mask):
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)
    
    def project(self, x):
        return self.projection_layer(x)
    
def build_transformer(
    src_vocab_size, tgt_vocab_size, src_seq, tgt_seq, 
    dim=512, N=6, n_head=8, dropout=0.1, d_ff=2048
):
    src_embed = InputEmbeddings(dim, src_vocab_size)
    tgt_embed = InputEmbeddings(dim, tgt_vocab_size)

    src_pos = PositionEmbedding(dim, src_seq)
    tgt_pos = PositionEmbedding(dim, tgt_seq)

    encoder_layers =[]
    for _ in range(N):
        encoder_self_attention_block = MultiHeadAttention(dim, n_head, dropout)
        feed_forward_block = FeedForwardBlock(dim, d_ff, dropout)
        encoder_block = EncoderBlock(dim, encoder_self_attention_block, feed_forward_block, dropout)
        encoder_layers.append(encoder_block)

    decoder_layers =[]
    for _ in range(N):
        decoder_self_attention_block = MultiHeadAttention(dim, n_head, dropout)
        decoder_cross_attention_block = MultiHeadAttention(dim, n_head, dropout)
        feed_forward_block = FeedForwardBlock(dim, d_ff, dropout)
        decoder_block = DecoderBlock(
            dim, decoder_self_attention_block, 
            decoder_cross_attention_block, feed_forward_block,
            dropout
        )
        decoder_layers.append(decoder_block)

    encoder = Encoder(dim, encoder_layers)
    decoder = Decoder(dim, decoder_layers)

    projection_layer = ProjectionLayer(dim, tgt_vocab_size)

    transformer = Transformer(
        encoder, decoder, src_embed, tgt_embed,
        src_pos, tgt_pos, projection_layer
    )

    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    
    return transformer