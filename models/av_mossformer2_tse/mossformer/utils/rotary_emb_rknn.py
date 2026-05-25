"""RotaryEmbedding without einsum (RKNN-friendly), API-compatible with rotary_embedding_torch."""

from __future__ import annotations

import torch
from einops import repeat
from rotary_embedding_torch import RotaryEmbedding as _RotaryEmbeddingBase


class RotaryEmbedding(_RotaryEmbeddingBase):
    """Same as rotary_embedding_torch.RotaryEmbedding but forward() uses broadcast, not einsum."""

    def forward(self, t: torch.Tensor, seq_len: int | None = None, offset: int = 0):
        should_cache = (
            self.cache_if_possible
            and not self.learned_freq
            and seq_len is not None
            and self.freqs_for != "pixel"
            and (offset + seq_len) <= self.cache_max_seq_len
        )

        if (
            should_cache
            and self.cached_freqs is not None
            and (offset + seq_len) <= self.cached_freqs_seq_len.item()
        ):
            return self.cached_freqs[offset : (offset + seq_len)].detach()

        freqs = self.freqs
        t_cast = t.type(freqs.dtype)
        # Equivalent to einsum('..., f -> ... f', t, freqs)
        freqs_out = t_cast.unsqueeze(-1) * freqs
        freqs_out = repeat(freqs_out, "... n -> ... (n r)", r=2)

        if should_cache and offset == 0:
            self.cached_freqs[:seq_len] = freqs_out.detach()
            self.cached_freqs_seq_len.copy_(seq_len)

        return freqs_out
