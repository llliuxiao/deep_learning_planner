import torch
import torch.nn.functional as F
from torch import nn, einsum, Tensor

from typing import List, Optional, Callable, Tuple
from beartype import beartype

from einops import pack, unpack, repeat, reduce, rearrange
from einops.layers.torch import Rearrange, Reduce


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
    def __init__(self, frame=6):
        super().__init__()
        self.frame = frame
        self.laser_pre = nn.Sequential(
            nn.AvgPool1d(kernel_size=3, stride=3),
            nn.Linear(360, 256)
        )
        self.global_plan_pre = nn.Linear(3, 256)
        self.laser_transformer = Transformer(dim=256, dim_head=64, heads=4, depth=6, attn_dropout=0, ff_dropout=0)
        self.global_plan_transformer = Transformer(dim=256, dim_head=64, heads=4, depth=4, attn_dropout=0, ff_dropout=0)
        self.dense = nn.Sequential(
            nn.Linear(512 + 3, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 2)
        )

    def forward(self, laser, global_plan, goal, laser_mask):
        pooled_laser = self.laser_pre(laser)
        high_dim_global_plan = self.global_plan_pre(global_plan)

        laser_mask = torch.squeeze(laser_mask)
        attended_laser = self.laser_transformer(pooled_laser, attn_mask=laser_mask)
        attended_global_plan = self.global_plan_transformer(high_dim_global_plan)

        laser_token = reduce(attended_laser, 'b f d -> b d', 'mean')
        global_plan_token = reduce(attended_global_plan, 'b f d -> b d', 'mean')

        tensor = torch.concat((laser_token, global_plan_token, goal), dim=1)
        output = self.dense(tensor)
        return output
