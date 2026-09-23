"""Perceiver-Resampler video former, ported from VPP (yjguo/dp-calvin) so its
checkpoint's `Video_Former.*` weights load directly.

Source: vpp-hiva/policy_models/module/Video_Former.py (itself referenced from
https://github.com/dhansmair/flamingo-mini), trimmed to the `3d` variant only
(the checkpoint we target uses `use_Former: 3d`).
"""

from __future__ import annotations

import torch
from einops import einsum, rearrange, repeat
from einops_exts import rearrange_many
from torch import nn
import torch.nn.functional as F


def feed_forward_layer(dim: int, mult: int = 4, activation: str = "gelu") -> nn.Sequential:
    inner_dim = int(dim * mult)
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, inner_dim, bias=False),
        nn.GELU(),
        nn.Linear(inner_dim, dim, bias=False),
    )


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        use_cross_attn: bool = False,
        y_dim: int = 512,
        qkv_bias: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        attn_mask: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        # Registered as a non-persistent buffer (rather than a plain attribute
        # reassigned in forward()) so it moves with .to(device)/.cuda() and is
        # never mutated in place -- reassigning a plain attribute to a tensor
        # created inside an eval-time `torch.inference_mode()` call would
        # otherwise permanently poison it into an "inference tensor" that
        # crashes a later training-mode forward pass.
        self.register_buffer("attn_mask", attn_mask, persistent=False)
        self.use_cross_attn = use_cross_attn
        if self.use_cross_attn:
            self.y_kv = nn.Linear(y_dim, dim * 2, bias=qkv_bias)
            self.gate = nn.Parameter(torch.zeros([self.num_heads]))

    def forward(self, x: torch.Tensor, y: torch.Tensor | None = None) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        x = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.attn_drop.p if self.training else 0.0, attn_mask=self.attn_mask
        )

        if self.use_cross_attn:
            N_y = y.shape[1]
            y_kv = self.y_kv(y).reshape(B, N_y, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
            y_k, y_v = y_kv.unbind(0)
            y_out = F.scaled_dot_product_attention(q, y_k, y_v, dropout_p=self.attn_drop.p if self.training else 0.0)
            y_out = y_out * self.gate.tanh().view(1, -1, 1, 1)
            x = x + y_out

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class PerceiverAttentionLayer(nn.Module):
    def __init__(self, dim: int, dim_head: int = 64, heads: int = 8):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        self.dim_head = dim_head
        inner_dim = dim_head * heads

        self.norm_media = nn.LayerNorm(dim)
        self.norm_latents = nn.LayerNorm(dim)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(dim, inner_dim, bias=False)
        self.to_v = nn.Linear(dim, inner_dim, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, features: torch.Tensor, latents: torch.Tensor) -> torch.Tensor:
        """Latents cross-attend to visual `features`.

        features: (batch, n_features, dim); latents: (batch, n_latents, dim).
        """
        n_heads = self.heads
        n_batch, n_features, _ = features.shape
        n_queries = latents.shape[1]

        x = self.norm_media(features)
        latents = self.norm_latents(latents)

        q = self.to_q(latents)
        q = rearrange(q, "b q (h d) -> b h q d", h=n_heads)

        kv_input = torch.cat((x, latents), dim=-2)
        k = self.to_k(kv_input)
        v = self.to_v(kv_input)
        k, v = rearrange_many((k, v), "b f (h d) -> b h f d", h=n_heads)

        q = q * self.scale
        sim = einsum(q, k, "b h q d, b h f d -> b h q f")
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        alphas = sim.softmax(dim=-1)

        out = einsum(alphas, v, "b h q f, b h f v -> b h q v")
        out = rearrange(out, "b h q v -> b q (h v)")
        return self.to_out(out)


class TempAttentionLayer(nn.Module):
    """Kept for checkpoint/architecture parity with upstream; the `3d` Video_Former
    variant actually uses plain `Attention` with an attention mask for its temporal
    mixing step, not this class -- see `Video_Former_3D.forward`."""

    def __init__(self, dim: int, dim_head: int = 64, heads: int = 8):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        self.dim_head = dim_head
        inner_dim = dim_head * heads

        self.norm_media = nn.LayerNorm(dim)
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(dim, inner_dim, bias=False)
        self.to_v = nn.Linear(dim, inner_dim, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        n_heads = self.heads
        n_batch, n_features, _ = features.shape

        x = self.norm_media(features)
        q = self.to_q(x)
        q = rearrange(q, "b q (h d) -> b h q d", h=n_heads)
        k = self.to_k(x)
        v = self.to_v(x)
        k, v = rearrange_many((k, v), "b f (h d) -> b h f d", h=n_heads)

        q = q * self.scale
        sim = einsum(q, k, "b h q d, b h f d -> b h q f")
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        alphas = sim.softmax(dim=-1)

        out = einsum(alphas, v, "b h q f, b h f v -> b h q v")
        out = rearrange(out, "b h q v -> b q (h v)")
        return self.to_out(out)


class Video_Former_3D(nn.Module):
    """Perceiver Resampler with a temporal mixing attention layer."""

    def __init__(
        self,
        dim: int,
        depth: int,
        condition_dim: int = 1280,
        dim_head: int = 64,
        heads: int = 8,
        num_latents: int = 64,
        num_frame: int = 16,
        num_time_embeds: int = 4,
        ff_mult: int = 4,
        activation: str = "gelu",
        trainable: bool = True,
        use_temporal: bool = False,
    ):
        super().__init__()

        self.dim = dim
        self.num_queries = num_latents
        self.num_frame = num_frame
        self.condition_dim = condition_dim
        self.use_temporal = use_temporal

        self.goal_emb = nn.Sequential(
            nn.Linear(condition_dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim),
        )
        frame_seq_len = num_latents // num_frame
        self.latents = nn.Parameter(torch.randn(self.num_frame, frame_seq_len, dim))
        self.time_pos_emb = nn.Parameter(torch.randn(num_time_embeds, 1, dim))
        attn_mask = torch.ones((num_frame, num_frame))

        self.layers = nn.ModuleList([])
        if self.use_temporal:
            for _ in range(depth):
                self.layers.append(
                    nn.ModuleList(
                        [
                            PerceiverAttentionLayer(dim=dim, dim_head=dim_head, heads=heads),
                            Attention(dim, num_heads=heads, qkv_bias=True, use_cross_attn=False, y_dim=512, attn_mask=attn_mask),
                            feed_forward_layer(dim=dim, mult=ff_mult, activation=activation),
                        ]
                    )
                )
        else:
            for _ in range(depth):
                self.layers.append(
                    nn.ModuleList(
                        [
                            PerceiverAttentionLayer(dim=dim, dim_head=dim_head, heads=heads),
                            feed_forward_layer(dim=dim, mult=ff_mult, activation=activation),
                        ]
                    )
                )

        self.norm = nn.LayerNorm(dim)
        for param in self.parameters():
            param.requires_grad = trainable

    def forward(self, x_f: torch.Tensor, mask: torch.Tensor | None = None, extra: torch.Tensor | None = None) -> torch.Tensor:
        """Resample multi-frame visual embeddings into a fixed token set.

        x_f: (batch, n_frames, n_features, d_visual) -> (batch, num_queries, dim).
        """
        assert x_f.ndim == 4
        batch_size, max_length, _, _ = x_f.shape

        time_pos_emb = self.time_pos_emb[:max_length].unsqueeze(0).expand(batch_size, -1, -1, -1)
        if mask is not None:
            time_pos_emb = time_pos_emb * mask.unsqueeze(-1).unsqueeze(-1)

        x_f = self.goal_emb(x_f)
        if extra is not None:
            extra = repeat(extra, "b q d -> b T q d", T=max_length)
            x_f = torch.cat([x_f, extra], dim=2)
        x_f = x_f + time_pos_emb

        x_f = rearrange(x_f, "b T n d -> (b T) n d")
        x = repeat(self.latents, "T q d -> b T q d", b=batch_size)
        x = rearrange(x, "b T q d -> (b T) q d")

        if self.use_temporal:
            for attn, temp_attn, ffw in self.layers:
                x = x + attn(x_f, x)
                x = rearrange(x, "(b T) q d -> (b q) T d", b=batch_size)
                x = x + temp_attn(x)
                x = rearrange(x, "(b q) T d -> (b T) q d", b=batch_size)
                x = x + ffw(x)
        else:
            for attn, ffw in self.layers:
                x = x + attn(x_f, x)
                x = x + ffw(x)

        x = x.reshape(batch_size, -1, x.shape[1], x.shape[2])
        x = rearrange(x, "b T q d -> b (T q) d")
        assert x.shape == torch.Size([batch_size, self.num_queries, self.dim])
        return self.norm(x)
