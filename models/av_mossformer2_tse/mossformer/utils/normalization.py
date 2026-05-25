import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm(nn.Module):
    """LayerNorm wrapper used by MossFormer2 code."""

    def __init__(self, input_size=None, input_shape=None, eps=1e-5, elementwise_affine=True):
        super().__init__()
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if input_shape is not None:
            input_size = input_shape[2:]

        self.norm = torch.nn.LayerNorm(
            input_size, eps=self.eps, elementwise_affine=self.elementwise_affine
        )

    def forward(self, x):
        return self.norm(x)


class CLayerNorm(nn.LayerNorm):
    """Channel-wise layer normalization."""

    def forward(self, sample):
        if sample.dim() != 3:
            raise RuntimeError(f"{self.__class__.__name__} only accepts 3-D tensor as input")
        # [N, C, T] -> [N, T, C]
        sample = torch.transpose(sample, 1, 2)
        sample = super().forward(sample)
        # [N, T, C] -> [N, C, T]
        sample = torch.transpose(sample, 1, 2)
        return sample


class ScaleNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.dim = dim
        self.scale = dim**-0.5
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1))
        self._rknn_safe = False
        self.register_buffer("_rknn_g_weight", torch.zeros(0), persistent=False)

    def enable_rknn_safe(self):
        """Replace scalar Mul(* self.g) with depthwise Conv1d for RKNN export."""
        self._rknn_safe = True
        with torch.no_grad():
            self._rknn_g_weight = self.g.item() * torch.ones(self.dim, 1, 1)

    def forward(self, x):
        norm = torch.norm(x, dim=-1, keepdim=True) * self.scale
        out = x / norm.clamp(min=self.eps)
        if not self._rknn_safe:
            return out * self.g
        # RKNN-safe: depthwise conv1d instead of scalar * self.g
        # out is [B, T, C]; conv1d needs [B, C, T]
        out_t = out.transpose(1, 2)
        out_t = F.conv1d(out_t, self._rknn_g_weight, groups=self.dim)
        return out_t.transpose(1, 2)

