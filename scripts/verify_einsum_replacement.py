#!/usr/bin/env python3
"""Verify RKNN-friendly op replacements match original PyTorch (fp32, rtol=atol=0)."""

from __future__ import annotations

import os
import sys

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import einsum

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from models.av_mossformer2_tse.mossformer.utils.Transformer import (  # noqa: E402
    FLASH_ShareA_FFConvM,
    OffsetScale,
    _shift_left_one,
)
from models.av_mossformer2_tse.mossformer.utils.rotary_emb_rknn import (  # noqa: E402
    RotaryEmbedding as RotaryEmbeddingRKNN,
)
from models.av_mossformer2_tse.av_mossformer2 import Decoder  # noqa: E402
from models.av_mossformer2_tse.mossformer.utils.one_path_flash_fsmn import (  # noqa: E402
    ScaledSinuEmbedding,
)


def _assert_close(a: torch.Tensor, b: torch.Tensor, name: str) -> None:
    torch.testing.assert_close(a, b, rtol=0, atol=0, msg=name)


def test_offset_scale() -> None:
    torch.manual_seed(0)
    m = OffsetScale(128, heads=4)
    x = torch.randn(2, 17, 128)
    ref = einsum("... d, h d -> ... h d", x, m.gamma) + m.beta
    ref = ref.unbind(dim=-2)
    got = m(x)
    for i, (r, g) in enumerate(zip(ref, got)):
        _assert_close(g, r, f"OffsetScale head {i}")


def _cal_attention_einsum(
    flash: FLASH_ShareA_FFConvM,
    x,
    quad_q,
    lin_q,
    quad_k,
    lin_k,
    v,
    u,
    mask=None,
):
    """Reference copy of cal_attention before explicit-op replacement."""
    from models.av_mossformer2_tse.mossformer.utils.Transformer import (
        default,
        exists,
        padding_to_multiple_of,
    )

    b, n, device, g = x.shape[0], x.shape[-2], x.device, flash.group_size
    if exists(mask):
        lin_mask = rearrange(mask, "... -> ... 1")
        lin_k = lin_k.masked_fill(~lin_mask, 0.0)

    if exists(flash.rotary_pos_emb):
        quad_q, lin_q, quad_k, lin_k = map(
            flash.rotary_pos_emb.rotate_queries_or_keys,
            (quad_q, lin_q, quad_k, lin_k),
        )

    padding = padding_to_multiple_of(n, g)
    if padding > 0:
        quad_q, quad_k, lin_q, lin_k, v, u = map(
            lambda t: torch.nn.functional.pad(t, (0, 0, 0, padding), value=0.0),
            (quad_q, quad_k, lin_q, lin_k, v, u),
        )
        mask = default(mask, torch.ones((b, n), device=device, dtype=torch.bool))
        mask = torch.nn.functional.pad(mask, (0, padding), value=False)

    quad_q, quad_k, lin_q, lin_k, v, u = map(
        lambda t: rearrange(t, "b (g n) d -> b g n d", n=flash.group_size),
        (quad_q, quad_k, lin_q, lin_k, v, u),
    )

    if exists(mask):
        mask = rearrange(mask, "b (g j) -> b g 1 j", j=g)

    sim = einsum("... i d, ... j d -> ... i j", quad_q, quad_k) / g
    attn = torch.nn.functional.relu(sim) ** 2
    attn = flash.dropout(attn)

    if exists(mask):
        attn = attn.masked_fill(~mask, 0.0)

    if flash.causal:
        causal_mask = torch.ones((g, g), dtype=torch.bool, device=device).triu(1)
        attn = attn.masked_fill(causal_mask, 0.0)

    quad_out_v = einsum("... i j, ... j d -> ... i d", attn, v)
    quad_out_u = einsum("... i j, ... j d -> ... i d", attn, u)

    if flash.causal:
        lin_kv = einsum("b g n d, b g n e -> b g d e", lin_k, v) / g
        lin_kv = lin_kv.cumsum(dim=1)
        lin_kv = torch.nn.functional.pad(lin_kv, (0, 0, 0, 0, 1, -1), value=0.0)
        lin_out_v = einsum("b g d e, b g n d -> b g n e", lin_kv, lin_q)

        lin_ku = einsum("b g n d, b g n e -> b g d e", lin_k, u) / g
        lin_ku = lin_ku.cumsum(dim=1)
        lin_ku = torch.nn.functional.pad(lin_ku, (0, 0, 0, 0, 1, -1), value=0.0)
        lin_out_u = einsum("b g d e, b g n d -> b g n e", lin_ku, lin_q)
    else:
        lin_kv = einsum("b g n d, b g n e -> b d e", lin_k, v) / n
        lin_out_v = einsum("b g n d, b d e -> b g n e", lin_q, lin_kv)

        lin_ku = einsum("b g n d, b g n e -> b d e", lin_k, u) / n
        lin_out_u = einsum("b g n d, b d e -> b g n e", lin_q, lin_ku)

    return tuple(
        map(
            lambda t: rearrange(t, "b g n d -> b (g n) d")[:, :n],
            (quad_out_v + lin_out_v, quad_out_u + lin_out_u),
        )
    )


def _make_flash_inputs(b=1, n=256, dim=64, qk_dim=32, causal=False):
    g = 256
    flash = FLASH_ShareA_FFConvM(
        dim=dim,
        group_size=g,
        query_key_dim=qk_dim,
        causal=causal,
        dropout=0.0,
        shift_tokens=False,
    )
    flash.eval()
    x = torch.randn(b, n, dim)
    quad_q = torch.randn(b, n, qk_dim)
    lin_q = torch.randn(b, n, qk_dim)
    quad_k = torch.randn(b, n, qk_dim)
    lin_k = torch.randn(b, n, qk_dim)
    v = torch.randn(b, n, dim // 2)
    u = torch.randn(b, n, dim // 2)
    return flash, x, quad_q, lin_q, quad_k, lin_k, v, u


def test_cal_attention(causal: bool) -> None:
    torch.manual_seed(1 if causal else 2)
    flash, x, quad_q, lin_q, quad_k, lin_k, v, u = _make_flash_inputs(causal=causal)
    ref = _cal_attention_einsum(flash, x, quad_q, lin_q, quad_k, lin_k, v, u)
    got = flash.cal_attention(x, quad_q, lin_q, quad_k, lin_k, v, u)
    _assert_close(got[0], ref[0], f"cal_attention causal={causal} out_v")
    _assert_close(got[1], ref[1], f"cal_attention causal={causal} out_u")


def test_scaled_sinu() -> None:
    torch.manual_seed(3)
    m = ScaledSinuEmbedding(32)
    x = torch.randn(1, 50, 64)
    n, device = x.shape[1], x.device
    t = torch.arange(n, device=device).type_as(m.inv_freq)
    ref = einsum("i , j -> i j", t, m.inv_freq)
    ref = torch.cat((ref.sin(), ref.cos()), dim=-1) * m.scale
    got = m(x)
    _assert_close(got, ref, "ScaledSinuEmbedding")


def test_shift_left_one() -> None:
    torch.manual_seed(10)
    x = torch.randn(2, 11, 8)
    ref = F.pad(x, (0, 0, 1, -1), value=0.0)
    got = _shift_left_one(x, dim=1)
    _assert_close(got, ref, "shift_left_one dim=1")
    x4 = torch.randn(1, 4, 6, 7)
    ref4 = F.pad(x4, (0, 0, 0, 0, 1, -1), value=0.0)
    got4 = _shift_left_one(x4, dim=1)
    _assert_close(got4, ref4, "shift_left_one dim=1 4d")


def test_rotary_embedding() -> None:
    from rotary_embedding_torch import RotaryEmbedding as RotaryOrig

    torch.manual_seed(11)
    dim = 32
    ref_mod = RotaryOrig(dim=dim)
    got_mod = RotaryEmbeddingRKNN(dim=dim)
    got_mod.load_state_dict(ref_mod.state_dict(), strict=False)
    q = torch.randn(2, 64, 48)
    ref = ref_mod.rotate_queries_or_keys(q)
    got = got_mod.rotate_queries_or_keys(q)
    _assert_close(got, ref, "RotaryEmbedding.rotate_queries_or_keys")


def test_decoder_ola() -> None:
    from types import SimpleNamespace

    torch.manual_seed(12)
    L, N = 16, 64
    args = SimpleNamespace()
    dec = Decoder(args, N, L)
    dec.eval()
    t_frames = 120
    dec.set_fixed_ola_frames(t_frames)
    mixture_w = torch.randn(1, N, t_frames)
    est_mask = torch.randn(1, N, t_frames)
    with torch.no_grad():
        est = mixture_w * est_mask
        est = est.transpose(2, 1)
        est = dec.basis_signals(est)
        x_bcl = est.transpose(1, 2)
        ref = dec.ola_conv(x_bcl).squeeze(1)
        got = dec._ola_synthesis(x_bcl)
    _assert_close(got, ref, "Decoder scatter OLA vs ConvTranspose1d")


def test_decoder_ola_gather_add() -> None:
    from types import SimpleNamespace

    torch.manual_seed(13)
    L, N = 16, 64
    args = SimpleNamespace()
    dec = Decoder(args, N, L)
    dec.eval()
    t_frames = 120
    dec.set_fixed_ola_frames(t_frames)
    dec.set_ola_export_mode("gather_add", chunk_size=256)
    x_bcl = torch.randn(1, L, t_frames)
    with torch.no_grad():
        x_flat = x_bcl.permute(0, 2, 1).reshape(1, t_frames * L)
        ref_mode = "scatter"
        dec._ola_export_mode = ref_mode
        ref = dec._ola_synthesis(x_bcl)
        dec.set_ola_export_mode("gather_add", chunk_size=256)
        got = dec._ola_synthesis(x_bcl)
    _assert_close(got, ref, "Decoder gather_add MatMul OLA vs scatter")


def test_full_model() -> None:
    ckpt = os.path.join(ROOT, "checkpoints", "AV_Mossformer", "last_best_weights_only.pt")
    cfg_path = os.path.join(ROOT, "checkpoints", "AV_Mossformer", "config.yaml")
    if not os.path.isfile(ckpt):
        print(f"[skip] full model: checkpoint not found at {ckpt}")
        return

    from export_onnx import build_model, compute_stream_window_lengths, load_config

    torch.manual_seed(42)
    cfg = load_config(cfg_path)
    model = build_model(cfg, ckpt)
    model.eval()

    t_audio, t_ref = compute_stream_window_lengths(
        audio_sr=16000, ref_sr=30.0, context_ms=100.0, infer_chunk_ms=500.0
    )
    mixture = torch.randn(1, t_audio)
    ref_in = torch.randn(1, t_ref, 96, 96, 3)

    with torch.no_grad():
        out = model(mixture, ref_in)
    if out.dim() == 3:
        assert out.shape == (1, 1, t_audio), f"unexpected output shape {out.shape}"
    else:
        assert out.shape == (1, t_audio), f"unexpected output shape {out.shape}"
    print(f"[ok] full model forward shape={tuple(out.shape)} (module-level einsum checks passed)")


def main() -> int:
    test_offset_scale()
    print("[ok] OffsetScale")
    test_cal_attention(causal=True)
    print("[ok] cal_attention causal=True")
    test_cal_attention(causal=False)
    print("[ok] cal_attention causal=False")
    test_scaled_sinu()
    print("[ok] ScaledSinuEmbedding")
    test_shift_left_one()
    print("[ok] shift_left_one (replaces F.pad 1,-1)")
    test_rotary_embedding()
    print("[ok] RotaryEmbedding RKNN")
    test_decoder_ola()
    print("[ok] Decoder overlap-add")
    test_decoder_ola_gather_add()
    print("[ok] Decoder gather_add MatMul OLA")
    test_full_model()
    print("All RKNN-friendly op replacement checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
