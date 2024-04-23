import math
from typing import Optional, Callable, Tuple

import torch
import torch.nn.functional as F
from beartype import beartype
from einops import reduce, rearrange
from torch import nn, einsum


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.register_buffer("beta", torch.zeros(dim))

    def forward(self, x):
        return F.layer_norm(x, x.shape[-1:], self.gamma, self.beta)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_len=20):
        super().__init__()

        # Compute the positional encoding once
        pos_enc = torch.zeros(max_seq_len, d_model)
        pos = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pos_enc[:, 0::2] = torch.sin(pos * div_term)
        pos_enc[:, 1::2] = torch.cos(pos * div_term)
        pos_enc = pos_enc.unsqueeze(0)

        # Register the positional encoding as a buffer to avoid it being
        # considered a parameter when saving the model
        self.register_buffer('pos_enc', pos_enc)

    def forward(self, x):
        x = x + self.pos_enc[:, :x.size(1), :]
        return x


class FeedForward(nn.Module):
    def __init__(self, dim, mult=4, dropout=0.):
        super().__init__()
        inner_dim = int(dim * mult)
        self.norm = LayerNorm(dim)

        self.net = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, cond_fn=None):
        x = self.norm(x)

        if exists(cond_fn):
            # adaptive layernorm
            x = cond_fn(x)

        return self.net(x)


class TransformerAttention(nn.Module):
    def __init__(
            self,
            dim,
            causal=False,
            dim_head=64,
            dim_context=None,
            heads=8,
            norm_context=False,
            dropout=0.1
    ):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.causal = causal
        inner_dim = dim_head * heads

        dim_context = default(dim_context, dim)

        self.norm = LayerNorm(dim)
        self.context_norm = LayerNorm(dim_context) if norm_context else nn.Identity()

        self.attn_dropout = nn.Dropout(dropout)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim_context, dim_head * 2, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim, bias=False),
            nn.Dropout(dropout)
        )

    def forward(
            self,
            x,
            context=None,
            mask=None,
            attn_bias=None,
            attn_mask=None,
            cond_fn: Optional[Callable] = None
    ):
        b = x.shape[0]

        if exists(context):
            context = self.context_norm(context)

        kv_input = default(context, x)

        x = self.norm(x)

        if exists(cond_fn):
            # adaptive layer-norm
            x = cond_fn(x)

        # noinspection PyTupleAssignmentBalance
        q, k, v = self.to_q(x), *self.to_kv(kv_input).chunk(2, dim=-1)

        q = rearrange(q, 'b n (h d) -> b h n d', h=self.heads)

        q = q * self.scale

        sim = einsum('b h i d, b j d -> b h i j', q, k)

        if exists(attn_bias):
            sim = sim + attn_bias

        if exists(attn_mask):
            sim = sim.masked_fill(attn_mask, -torch.finfo(sim.dtype).max)

        if exists(mask):
            mask = rearrange(mask, 'b j -> b 1 1 j')
            sim = sim.masked_fill(~mask, -torch.finfo(sim.dtype).max)

        if self.causal:
            i, j = sim.shape[-2:]
            causal_mask = torch.ones((i, j), dtype=torch.bool, device=x.device).triu(j - i + 1)
            sim = sim.masked_fill(causal_mask, -torch.finfo(sim.dtype).max)

        attn = sim.softmax(dim=-1)
        attn = self.attn_dropout(attn)

        out = einsum('b h i j, b j d -> b h i d', attn, v)

        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


@beartype
class Transformer(nn.Module):
    def __init__(
            self,
            dim,
            dim_head=64,
            heads=8,
            depth=6,
            attn_dropout=0.,
            ff_dropout=0.
    ):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                TransformerAttention(dim=dim, heads=heads, dropout=attn_dropout),
                FeedForward(dim=dim, dropout=ff_dropout)
            ]))

    def forward(
            self,
            x,
            cond_fns: Optional[Tuple[Callable, ...]] = None,
            attn_mask=None
    ):
        cond_fns = iter(default(cond_fns, []))

        for attn, ff in self.layers:
            x = attn(x, attn_mask=attn_mask, cond_fn=next(cond_fns, None)) + x
            x = ff(x, cond_fn=next(cond_fns, None)) + x
        return x


@beartype
class RobotTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        # laser
        self.laser_pre = nn.Linear(1080, 512)
        self.laser_position_encoding = PositionalEncoding(d_model=512, max_seq_len=6)
        self.laser_transformer = Transformer(dim=512, dim_head=64,
                                             heads=8, depth=6,
                                             attn_dropout=0.1, ff_dropout=0.1)

        # global plan
        self.global_plan_pre = nn.Linear(3, 256)
        self.global_plan_position_encoding = PositionalEncoding(d_model=256, max_seq_len=20)
        self.global_plan_transformer = Transformer(dim=256, dim_head=64,
                                                   heads=4, depth=4,
                                                   attn_dropout=0.1, ff_dropout=0.1)
        self.dense = nn.Sequential(
            nn.Linear(512 + 256 + 3, 1024), nn.ReLU(),
            nn.Linear(1024, 512), nn.ReLU()
        )
        self.mlp_extractor_policy = nn.Sequential(nn.Linear(512, 256), nn.ReLU())
        self.policy_net = nn.Sequential(nn.Linear(256, 2))

    def forward(self, laser, global_plan, goal, laser_mask):
        pooled_laser = self.laser_pre(laser)
        positional_laser = self.laser_position_encoding(pooled_laser)
        laser_mask = torch.squeeze(laser_mask)
        attended_laser = self.laser_transformer(positional_laser, attn_mask=laser_mask)

        high_dim_global_plan = self.global_plan_pre(global_plan)
        positional_path = self.global_plan_position_encoding(high_dim_global_plan)
        attended_global_plan = self.global_plan_transformer(positional_path)

        laser_token = reduce(attended_laser, 'b f d -> b d', 'mean')
        global_plan_token = reduce(attended_global_plan, 'b f d -> b d', 'mean')

        tensor = torch.concat((laser_token, global_plan_token, goal), dim=1)
        feature = self.dense(tensor)
        return self.policy_net(self.mlp_extractor_policy(feature))
