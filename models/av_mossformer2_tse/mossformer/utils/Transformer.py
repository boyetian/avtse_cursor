"""Transformer implementation for MossFormer2 (as used by ClearerVoice-Studio).

This file is vendored into target_speaker_extraction_online to avoid a runtime dependency
on the sibling ClearerVoice-Studio repo.
"""

import math
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from .rotary_emb_rknn import RotaryEmbedding

from .fsmn import UniDeepFsmn, UniDeepFsmn_dilated
from .normalization import LayerNorm, CLayerNorm, ScaleNorm
from .conv_module import ConvModule


def exists(val):
    return val is not None


def padding_to_multiple_of(n, mult):
    remainder = n % mult
    if remainder == 0:
        return 0
    return mult - remainder


def default(val, d):
    return val if exists(val) else d


def _shift_left_one(x: torch.Tensor, dim: int) -> torch.Tensor:
    """Equivalent to F.pad(..., 1, -1) on dimension dim (left pad 1 zero, drop last)."""
    pad_shape = list(x.shape)
    pad_shape[dim] = 1
    head = x.new_zeros(pad_shape)
    return torch.cat([head, x.narrow(dim, 0, x.size(dim) - 1)], dim=dim)


def _pad_right_time(x: torch.Tensor, padding: int) -> torch.Tensor:
    """Equivalent to F.pad(x, (0, 0, 0, padding)) on [b, n, d] (RKNN-friendly)."""
    if padding <= 0:
        return x
    tail = x.new_zeros(x.size(0), padding, x.size(-1))
    return torch.cat([x, tail], dim=-2)


def _pad_right_mask(mask: torch.Tensor, padding: int) -> torch.Tensor:
    """Equivalent to F.pad(mask, (0, padding), value=False) on [b, n]."""
    if padding <= 0:
        return mask
    tail = mask.new_zeros(mask.size(0), padding, dtype=torch.bool)
    return torch.cat([mask, tail], dim=-1)


class FFConvM(nn.Module):
    def __init__(self, dim_in, dim_out, norm_klass=nn.LayerNorm, dropout=0.1, causal: bool = False):
        super().__init__()
        self.mdl = nn.Sequential(
            norm_klass(dim_in),
            nn.Linear(dim_in, dim_out),
            nn.SiLU(),
            ConvModule(dim_out, causal=causal),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.mdl(x)


class Gated_FSMN_dilated(nn.Module):
    def __init__(self, in_channels, out_channels, lorder, hidden_size, causal: bool = False):
        super().__init__()
        self.to_u = FFConvM(
            dim_in=in_channels, dim_out=hidden_size, norm_klass=nn.LayerNorm, dropout=0.1, causal=causal
        )
        self.to_v = FFConvM(
            dim_in=in_channels, dim_out=hidden_size, norm_klass=nn.LayerNorm, dropout=0.1, causal=causal
        )
        self.fsmn = UniDeepFsmn_dilated(in_channels, out_channels, lorder, hidden_size, causal=causal)

    def forward(self, x):
        input = x
        x_u = self.to_u(x)
        x_v = self.to_v(x)
        x_u = self.fsmn(x_u)
        x = x_v * x_u + input
        return x


class Gated_FSMN_Block_Dilated(nn.Module):
    def __init__(
        self,
        dim,
        inner_channels=256,
        group_size=256,
        norm_type="scalenorm",
        causal: bool = False,
    ):
        super(Gated_FSMN_Block_Dilated, self).__init__()
        if norm_type == "scalenorm":
            norm_klass = ScaleNorm
        elif norm_type == "layernorm":
            norm_klass = nn.LayerNorm
        else:
            norm_klass = ScaleNorm

        self.group_size = group_size

        self.conv1 = nn.Sequential(
            nn.Conv1d(dim, inner_channels, kernel_size=1),
            nn.PReLU(),
        )
        self.norm1 = CLayerNorm(inner_channels)
        self.gated_fsmn = Gated_FSMN_dilated(
            inner_channels, inner_channels, lorder=20, hidden_size=inner_channels, causal=causal
        )
        self.norm2 = CLayerNorm(inner_channels)
        self.conv2 = nn.Conv1d(inner_channels, dim, kernel_size=1)

    def forward(self, input):
        conv1 = self.conv1(input.transpose(2, 1))
        norm1 = self.norm1(conv1)
        seq_out = self.gated_fsmn(norm1.transpose(2, 1))
        norm2 = self.norm2(seq_out.transpose(2, 1))
        conv2 = self.conv2(norm2)
        return conv2.transpose(2, 1) + input


class OffsetScale(nn.Module):
    def __init__(self, dim, heads=1):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(heads, dim))
        self.beta = nn.Parameter(torch.zeros(heads, dim))
        nn.init.normal_(self.gamma, std=0.02)

    def forward(self, x):
        # Broadcast mul/add instead of ellipsis einsum (RKNN fold_constant / ORT shape infer).
        out = x.unsqueeze(-2) * self.gamma + self.beta
        return out.unbind(dim=-2)


class FLASH_ShareA_FFConvM(nn.Module):
    def __init__(
        self,
        *,
        dim,
        group_size=256,
        query_key_dim=128,
        expansion_factor=1.0,
        causal=False,
        dropout=0.1,
        rotary_pos_emb=None,
        norm_klass=nn.LayerNorm,
        shift_tokens=True,
    ):
        super().__init__()
        hidden_dim = int(dim * expansion_factor)
        self.group_size = group_size
        self.causal = causal
        self.shift_tokens = shift_tokens
        self.rotary_pos_emb = rotary_pos_emb
        self.dropout = nn.Dropout(dropout)
        self._rknn_safe = False
        self.register_buffer("_rknn_inv_g_weight", torch.zeros(0), persistent=False)

        self.to_hidden = FFConvM(
            dim_in=dim, dim_out=hidden_dim, norm_klass=norm_klass, dropout=dropout, causal=causal
        )
        self.to_qk = FFConvM(
            dim_in=dim, dim_out=query_key_dim, norm_klass=norm_klass, dropout=dropout, causal=causal
        )
        self.qk_offset_scale = OffsetScale(query_key_dim, heads=4)
        self.to_out = FFConvM(
            dim_in=dim * 2, dim_out=dim, norm_klass=norm_klass, dropout=dropout, causal=causal
        )
        self.gateActivate = nn.Sigmoid()

    def enable_rknn_safe(self, num_groups: int):
        """Replace scalar Div(/ g) with depthwise Conv2d for RKNN export."""
        self._rknn_safe = True
        with torch.no_grad():
            self._rknn_inv_g_weight = (1.0 / self.group_size) * torch.ones(num_groups, 1, 1, 1)

    def forward(self, x, mask=None):
        normed_x = x
        if self.shift_tokens:
            x_shift, x_pass = normed_x.chunk(2, dim=-1)
            x_shift = _shift_left_one(x_shift, dim=1)
            normed_x = torch.cat((x_shift, x_pass), dim=-1)

        v, u = self.to_hidden(normed_x).chunk(2, dim=-1)
        qk = self.to_qk(normed_x)
        quad_q, lin_q, quad_k, lin_k = self.qk_offset_scale(qk)
        att_v, att_u = self.cal_attention(x, quad_q, lin_q, quad_k, lin_k, v, u, mask=mask)
        out = (att_u * v) * self.gateActivate(att_v * u)
        x = x + self.to_out(out)
        return x

    def cal_attention(self, x, quad_q, lin_q, quad_k, lin_k, v, u, mask=None):
        b, n, device, g = x.shape[0], x.shape[-2], x.device, self.group_size
        if exists(mask):
            lin_mask = rearrange(mask, "... -> ... 1")
            lin_k = lin_k.masked_fill(~lin_mask, 0.0)

        if exists(self.rotary_pos_emb):
            quad_q, lin_q, quad_k, lin_k = map(
                self.rotary_pos_emb.rotate_queries_or_keys, (quad_q, lin_q, quad_k, lin_k)
            )

        padding = padding_to_multiple_of(n, g)
        if padding > 0:
            quad_q, quad_k, lin_q, lin_k, v, u = map(
                lambda t: _pad_right_time(t, padding), (quad_q, quad_k, lin_q, lin_k, v, u)
            )
            mask = default(mask, torch.ones((b, n), device=device, dtype=torch.bool))
            mask = _pad_right_mask(mask, padding)

        quad_q, quad_k, lin_q, lin_k, v, u = map(
            lambda t: rearrange(t, "b (g n) d -> b g n d", n=self.group_size),
            (quad_q, quad_k, lin_q, lin_k, v, u),
        )

        if exists(mask):
            mask = rearrange(mask, "b (g j) -> b g 1 j", j=g)

        sim = torch.matmul(quad_q, quad_k.transpose(-1, -2))
        if self._rknn_safe:
            sim = F.conv2d(sim, self._rknn_inv_g_weight, groups=self._rknn_inv_g_weight.shape[0])
        else:
            sim = sim / g
        attn = F.relu(sim)
        if self._rknn_safe:
            attn = attn * attn
        else:
            attn = attn ** 2
        attn = self.dropout(attn)

        if exists(mask):
            attn = attn.masked_fill(~mask, 0.0)

        if self.causal:
            causal_mask = torch.ones((g, g), dtype=torch.bool, device=device).triu(1)
            attn = attn * (~causal_mask).to(dtype=attn.dtype)

        quad_out_v = torch.matmul(attn, v)
        quad_out_u = torch.matmul(attn, u)

        if self.causal:
            lin_kv = torch.matmul(lin_k.transpose(-2, -1), v)
            if self._rknn_safe:
                lin_kv = F.conv2d(lin_kv, self._rknn_inv_g_weight, groups=self._rknn_inv_g_weight.shape[0])
            else:
                lin_kv = lin_kv / g
            lin_kv = lin_kv.cumsum(dim=1)
            lin_kv = _shift_left_one(lin_kv, dim=1)
            lin_out_v = torch.matmul(lin_q, lin_kv)

            lin_ku = torch.matmul(lin_k.transpose(-2, -1), u)
            if self._rknn_safe:
                lin_ku = F.conv2d(lin_ku, self._rknn_inv_g_weight, groups=self._rknn_inv_g_weight.shape[0])
            else:
                lin_ku = lin_ku / g
            lin_ku = lin_ku.cumsum(dim=1)
            lin_ku = _shift_left_one(lin_ku, dim=1)
            lin_out_u = torch.matmul(lin_q, lin_ku)
        else:
            lin_k_flat = rearrange(lin_k, "b g n d -> b (g n) d")
            v_flat = rearrange(v, "b g n e -> b (g n) e")
            lin_kv = torch.matmul(lin_k_flat.transpose(-1, -2), v_flat) / n
            lin_out_v = torch.matmul(lin_q, lin_kv)

            u_flat = rearrange(u, "b g n e -> b (g n) e")
            lin_ku = torch.matmul(lin_k_flat.transpose(-1, -2), u_flat) / n
            lin_out_u = torch.matmul(lin_q, lin_ku)

        return tuple(
            map(
                lambda t: rearrange(t, "b g n d -> b (g n) d")[:, :n],
                (quad_out_v + lin_out_v, quad_out_u + lin_out_u),
            )
        )


class FLASHTransformer_DualA_FSMN(nn.Module):
    def __init__(
        self,
        *,
        dim,
        depth,
        group_size=256,
        query_key_dim=128,
        expansion_factor=4.0,
        causal=False,
        attn_dropout=0.1,
        norm_type="scalenorm",
        shift_tokens=True,
        fsmn_inner_channels: int = 256,
    ):
        super().__init__()
        assert norm_type in ("scalenorm", "layernorm")
        norm_klass = ScaleNorm if norm_type == "scalenorm" else nn.LayerNorm

        self.group_size = group_size
        rotary_pos_emb = RotaryEmbedding(dim=min(32, query_key_dim))
        self.fsmn = nn.ModuleList(
            [
                Gated_FSMN_Block_Dilated(dim, inner_channels=int(fsmn_inner_channels), causal=causal)
                for _ in range(depth)
            ]
        )
        self.layers = nn.ModuleList(
            [
                FLASH_ShareA_FFConvM(
                    dim=dim,
                    group_size=group_size,
                    query_key_dim=query_key_dim,
                    expansion_factor=expansion_factor,
                    causal=causal,
                    dropout=attn_dropout,
                    rotary_pos_emb=rotary_pos_emb,
                    norm_klass=norm_klass,
                    shift_tokens=shift_tokens,
                )
                for _ in range(depth)
            ]
        )

    def forward(self, x, mask=None):
        for ii, flash in enumerate(self.layers):
            x = flash(x, mask=mask)
            x = self.fsmn[ii](x)
        return x


class TransformerEncoder_FLASH_DualA_FSMN(nn.Module):
    def __init__(
        self,
        num_layers,
        nhead,
        d_ffn,
        input_shape=None,
        d_model=None,
        kdim=None,
        vdim=None,
        dropout=0.0,
        activation=nn.ReLU,
        normalize_before=False,
        causal=False,
        attention_type="regularMHA",
        fsmn_inner_channels: int = 256,
    ):
        super().__init__()
        self.flashT = FLASHTransformer_DualA_FSMN(
            dim=d_model,
            depth=num_layers,
            causal=causal,
            fsmn_inner_channels=int(fsmn_inner_channels),
        )
        self.norm = LayerNorm(d_model, eps=1e-6)

    def forward(
        self,
        src,
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        pos_embs: Optional[torch.Tensor] = None,
    ):
        output = self.flashT(src)
        output = self.norm(output)
        return output

