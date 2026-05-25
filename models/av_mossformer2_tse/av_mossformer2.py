import math
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from .mossformer.utils.one_path_flash_fsmn import Dual_Path_Model, SBFLASHBlock_DualA
from .visual_frontend import Visual_encoder

EPS = 1e-8


def encoder_frame_count(audio_samples: int, kernel_size: int = 16) -> int:
    """Encoder 1D conv output length for fixed-length ONNX/RKNN export."""
    stride = max(1, kernel_size // 2)
    return (int(audio_samples) - int(kernel_size)) // stride + 1


class Mossformer(nn.Module):
    def __init__(self, args):
        super(Mossformer, self).__init__()
        self.args = args
        N, L = args.network_audio.encoder_out_nchannels, args.network_audio.encoder_kernel_size

        self.encoder = Encoder(L, N)
        self.separator = Separator(args)
        self.decoder = Decoder(args, N, L)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_normal_(p)

    def forward(self, mixture, visual):
        st = getattr(self.args, "infer_forward_timing", None)
        if st is not None:
            t0 = time.perf_counter()
        mixture_w = self.encoder(mixture)
        if st is not None:
            st["mossformer_encoder"] = float(st.get("mossformer_encoder", 0.0)) + (
                time.perf_counter() - t0
            )
            t1 = time.perf_counter()
        est_mask = self.separator(mixture_w, visual)
        if st is not None:
            st["mossformer_separator"] = float(st.get("mossformer_separator", 0.0)) + (
                time.perf_counter() - t1
            )
            t2 = time.perf_counter()
        est_source = self.decoder(mixture_w, est_mask)
        if st is not None:
            st["mossformer_decoder"] = float(st.get("mossformer_decoder", 0.0)) + (
                time.perf_counter() - t2
            )

        T_origin = mixture.size(-1)
        T_conv = est_source.size(-1)
        tail_pad = int(T_origin - T_conv)
        if tail_pad > 0:
            est_source = torch.cat(
                [est_source, est_source.new_zeros(est_source.size(0), tail_pad)], dim=-1
            )
        return est_source


class Encoder(nn.Module):
    def __init__(self, L, N):
        super(Encoder, self).__init__()
        self.L, self.N = L, N
        self.conv1d_U = nn.Conv1d(1, N, kernel_size=L, stride=L // 2, bias=False)

    def forward(self, mixture):
        mixture = torch.unsqueeze(mixture, 1)
        mixture_w = F.relu(self.conv1d_U(mixture))
        return mixture_w


def build_ola_scatter_indices(
    kernel_size: int, stride: int, t_frames: int
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Gather/scatter indices for OLA (nnz=T*L, ONNX-exportable, no sparse_coo_tensor)."""
    L = int(kernel_size)
    T = int(t_frames)
    stride = int(stride)
    out_len = (T - 1) * stride + L
    rows: list[int] = []
    cols: list[int] = []
    for t in range(T):
        base_col = t * stride
        base_row = t * L
        for l in range(L):
            c = base_col + l
            if c < out_len:
                rows.append(base_row + l)
                cols.append(c)
    return torch.tensor(rows, dtype=torch.long), torch.tensor(cols, dtype=torch.long), out_len


class Decoder(nn.Module):
    def __init__(self, args, N, L):
        super(Decoder, self).__init__()
        self.N, self.L, self.args = N, L, args
        self.stride = L // 2
        self.basis_signals = nn.Linear(N, L, bias=False)
        self.ola_conv = nn.ConvTranspose1d(L, 1, kernel_size=L, stride=self.stride, bias=False)
        with torch.no_grad():
            self.ola_conv.weight.copy_(torch.eye(L, dtype=torch.float32).view(L, 1, L))
        self.register_buffer("ola_row_idx", torch.zeros(0, dtype=torch.long), persistent=False)
        self.register_buffer("ola_col_idx", torch.zeros(0, dtype=torch.long), persistent=False)
        self._ola_out_len = 0
        self._ola_t_frames = 0

    def set_fixed_ola_frames(self, t_frames: int) -> None:
        """Pin OLA scatter indices for fixed-length export (small buffers, not dense MatMul weight)."""
        t_frames = int(t_frames)
        row_idx, col_idx, out_len = build_ola_scatter_indices(self.L, self.stride, t_frames)
        self.register_buffer("ola_row_idx", row_idx, persistent=False)
        self.register_buffer("ola_col_idx", col_idx, persistent=False)
        self._ola_out_len = int(out_len)
        self._ola_t_frames = t_frames

    def clear_fixed_ola_frames(self) -> None:
        """Use ConvTranspose OLA (RKNN-friendly, avoids ScatterElements from scatter_add)."""
        self.register_buffer("ola_row_idx", torch.zeros(0, dtype=torch.long), persistent=False)
        self.register_buffer("ola_col_idx", torch.zeros(0, dtype=torch.long), persistent=False)
        self._ola_out_len = 0
        self._ola_t_frames = 0

    def _ola_synthesis(self, x_bcl: torch.Tensor) -> torch.Tensor:
        b, L, t = x_bcl.shape
        if self.ola_row_idx.numel() > 0 and t == self._ola_t_frames:
            x_flat = x_bcl.permute(0, 2, 1).reshape(b, t * L)
            out = x_flat.new_zeros(b, self._ola_out_len)
            col = self.ola_col_idx.unsqueeze(0).expand(b, -1)
            contrib = x_flat[:, self.ola_row_idx]
            out.scatter_add_(1, col, contrib)
            return out
        return self.ola_conv(x_bcl).squeeze(1)

    def forward(self, mixture_w, est_mask):
        est_source = mixture_w * est_mask
        est_source = torch.transpose(est_source, 2, 1)
        est_source = self.basis_signals(est_source)
        est_source = self._ola_synthesis(est_source.transpose(1, 2))
        return est_source


class Separator(nn.Module):
    def __init__(self, args):
        super(Separator, self).__init__()

        self.layer_norm = nn.GroupNorm(1, args.network_audio.encoder_out_nchannels, eps=1e-8)
        N = args.network_audio.encoder_out_nchannels
        inner_nchannels = int(getattr(args.network_audio, "masknet_inner_nchannels", N))
        if inner_nchannels <= 0:
            raise ValueError(f"masknet_inner_nchannels must be > 0, got {inner_nchannels}")
        self.bottleneck_conv1x1 = nn.Conv1d(N, N, 1, bias=False)
        self.masknet_in_proj = nn.Conv1d(N, inner_nchannels, 1, bias=False)
        self.masknet_out_proj = nn.Conv1d(inner_nchannels, N, 1, bias=False)

        _causal = bool(int(getattr(args, "causal", 0) or 0))
        # FSMN gated block inner channels. Keep 256 default for ckpt compatibility.
        fsmn_inner_channels = int(getattr(args.network_audio, "fsmn_inner_channels", 256) or 256)
        intra_model = SBFLASHBlock_DualA(
            num_layers=args.network_audio.intra_numlayers,
            d_model=inner_nchannels,
            nhead=args.network_audio.intra_nhead,
            d_ffn=args.network_audio.intra_dffn,
            dropout=args.network_audio.intra_dropout,
            use_positional_encoding=args.network_audio.intra_use_positional,
            norm_before=args.network_audio.intra_norm_before,
            causal=_causal,
            fsmn_inner_channels=fsmn_inner_channels,
        )

        self.masknet = Dual_Path_Model(
            in_channels=inner_nchannels,
            out_channels=inner_nchannels,
            intra_model=intra_model,
            num_layers=args.network_audio.masknet_numlayers,
            norm=args.network_audio.masknet_norm,
            K=args.network_audio.masknet_chunksize,
            num_spks=args.network_audio.masknet_numspks,
            skip_around_intra=args.network_audio.masknet_extraskipconnection,
            linear_layer_after_inter_intra=args.network_audio.masknet_useextralinearlayer,
        )

        self.av_conv = nn.Conv1d(
            args.network_audio.encoder_out_nchannels + args.network_reference.emb_size,
            args.network_audio.encoder_out_nchannels,
            1,
            bias=True,
        )
        self.stream_cache_enable = bool(int(getattr(args.network_audio, "stream_cache_enable", 0) or 0))
        self.stream_block_frames = int(getattr(args.network_audio, "stream_block_frames", 0) or 0)
        self.stream_overlap_frames = int(getattr(args.network_audio, "stream_overlap_frames", 0) or 0)
        # Hop-gated separator cache + periodic full refresh.
        # - gate_every_hops=1: keep legacy behavior (try cache every hop).
        # - refresh_every_hops>0: force one full recompute every N hops to break drift.
        self.stream_cache_gate_every_hops = max(
            1, int(getattr(args.network_audio, "stream_cache_gate_every_hops", 1) or 1)
        )
        self.stream_cache_refresh_every_hops = max(
            0, int(getattr(args.network_audio, "stream_cache_refresh_every_hops", 0) or 0)
        )
        # Debug print per forward call:
        # 1) yaml: network_audio.stream_cache_debug: 1
        # 2) env: STREAM_CACHE_DEBUG=1 (overrides yaml)
        env_dbg = os.getenv("STREAM_CACHE_DEBUG")
        if env_dbg is None:
            self.stream_cache_debug = bool(int(getattr(args.network_audio, "stream_cache_debug", 0) or 0))
        else:
            self.stream_cache_debug = bool(int(env_dbg))
        self._stream_dbg_step = 0
        self._cached_mask = None
        self._cached_len = 0
        self._stream_hop_step = 0
        # Set to int encoder frame count before fixed-shape ONNX export (RKNN-friendly Resize).
        self.fixed_encoder_frames = None

    def clear_stream_cache(self):
        self._cached_mask = None
        self._cached_len = 0
        self._stream_dbg_step = 0
        self._stream_hop_step = 0

    def trim_stream_cache(self, n_drop_frames: int) -> int:
        """Drop n leading frames from cached separator output.
        Used by ring-buffer inference when input head is trimmed."""
        n = int(n_drop_frames)
        if n <= 0 or self._cached_mask is None:
            return 0
        cur = int(self._cached_len)
        n = min(n, cur)
        if n <= 0:
            return 0
        if n >= cur:
            self._cached_mask = None
            self._cached_len = 0
            return cur
        self._cached_mask = self._cached_mask[:, :, n:].contiguous()
        self._cached_len = cur - n
        return n

    def _run_masknet_path(self, x):
        x = self.masknet_in_proj(x)
        x = self.masknet(x)
        x = x.squeeze(0)
        x = self.masknet_out_proj(x)
        return x

    @staticmethod
    def _align_visual_length(visual, target_len: int):
        return F.interpolate(
            visual,
            size=(int(target_len),),
            mode="linear",
            align_corners=False,
        )

    def forward(self, x, visual):
        _, _, D = x.size()
        x = self.layer_norm(x)
        x = self.bottleneck_conv1x1(x)

        fixed_d = getattr(self, "fixed_encoder_frames", None)
        if fixed_d is not None:
            visual = self._align_visual_length(visual, fixed_d)
        else:
            visual = self._align_visual_length(visual, D)
        x = torch.cat((x, visual), 1)
        x = self.av_conv(x)

        if (
            self.stream_cache_enable
            and (not self.training)
            and self._cached_mask is not None
            and self._cached_mask.dim() == 3
            and int(self._cached_len) > 0
            and D < int(self._cached_len)
        ):
            self._cached_mask = self._cached_mask[:, :, :D]
            self._cached_len = int(D)

        cached_len = int(self._cached_len)
        new_frames = int(D - cached_len)
        hop_step = int(self._stream_hop_step)
        gate_ok = (hop_step % int(self.stream_cache_gate_every_hops)) == 0
        refresh_hit = (
            self.stream_cache_refresh_every_hops > 0
            and hop_step > 0
            and (hop_step % int(self.stream_cache_refresh_every_hops)) == 0
        )
        use_stream_cache = (
            self.stream_cache_enable
            and (not self.training)
            and self._cached_mask is not None
            and self._cached_mask.dim() == 3
            and cached_len > 0
            and D >= cached_len
            and self.stream_block_frames > 0
            and gate_ok
            and (not refresh_hit)
        )
        if self.stream_cache_debug:
            miss_reasons = []
            if not self.stream_cache_enable:
                miss_reasons.append("cache_disabled")
            if self.training:
                miss_reasons.append("training_mode")
            if self._cached_mask is None:
                miss_reasons.append("cached_mask_none")
            elif self._cached_mask.dim() != 3:
                miss_reasons.append(f"cached_mask_dim_{self._cached_mask.dim()}")
            if cached_len <= 0:
                miss_reasons.append("cached_len_le_0")
            if D < cached_len:
                miss_reasons.append(f"D_lt_cached({D}<{cached_len})")
            if self.stream_block_frames <= 0:
                miss_reasons.append("block_le_0")
            if not gate_ok:
                miss_reasons.append(f"gate_skip(step={hop_step},every={int(self.stream_cache_gate_every_hops)})")
            if refresh_hit:
                miss_reasons.append(
                    f"periodic_refresh(step={hop_step},every={int(self.stream_cache_refresh_every_hops)})"
                )
            miss_text = "none" if use_stream_cache else ",".join(miss_reasons)
            print(
                "[stream_cache] "
                f"step={self._stream_dbg_step} "
                f"hop_step={hop_step} "
                f"D={int(D)} cached_len={cached_len} "
                f"new_frames={new_frames} "
                f"use_stream_cache={int(use_stream_cache)} "
                f"cfg_block={int(self.stream_block_frames)} "
                f"cfg_overlap={int(self.stream_overlap_frames)} "
                f"cfg_gate_every={int(self.stream_cache_gate_every_hops)} "
                f"cfg_refresh_every={int(self.stream_cache_refresh_every_hops)} "
                f"miss={miss_text}"
            )
            self._stream_dbg_step += 1      

        if use_stream_cache:
            # Causal streaming fast-path: keep cached prefix and recompute only tail (block + overlap).
            # new_frames = D - int(self._cached_len)
            block = max(int(self.stream_block_frames), int(new_frames))
            overlap = max(0, int(self.stream_overlap_frames))
            start = max(0, D - (block + overlap))
            x_tail = x[:, :, start:]
            y_tail = self._run_masknet_path(x_tail)

            keep_from = max(0, D - block)
            keep_tail = D - keep_from
            if keep_from > 0 and self._cached_mask.size(2) >= keep_from:
                y = torch.cat([self._cached_mask[:, :, :keep_from], y_tail[:, :, -keep_tail:]], dim=2)
            else:
                y = y_tail[:, :, -keep_tail:]
        else:
            y = self._run_masknet_path(x)

        if self.stream_cache_enable and (not self.training):
            self._cached_mask = y.detach()
            self._cached_len = int(y.size(2))
        if not self.training:
            self._stream_hop_step = int(self._stream_hop_step) + 1
        return y


class av_mossformer2(nn.Module):
    def __init__(self, args):
        super(av_mossformer2, self).__init__()
        self.args = args
        self.sep_network = Mossformer(args)
        self.ref_encoder = Visual_encoder(args)
        na = getattr(args, "network_audio", None)
        stream_cache_enable = bool(int(getattr(na, "stream_cache_enable", 0) or 0))
        self.ref_stream_cache_enable = bool(
            int(getattr(na, "ref_stream_cache_enable", int(stream_cache_enable)) or 0)
        )
        self.ref_stream_block_frames = int(
            getattr(na, "ref_stream_block_frames", getattr(na, "stream_block_frames", 0)) or 0
        )
        self.ref_stream_overlap_frames = int(
            getattr(na, "ref_stream_overlap_frames", getattr(na, "stream_overlap_frames", 0)) or 0
        )
        self._cached_ref = None
        self._cached_ref_len = 0

    def forward(self, mixture, ref):
        st = getattr(self.args, "infer_forward_timing", None)
        t0 = time.perf_counter() if st is not None else 0.0
        ref_len = int(ref.size(1))
        use_ref_cache = False
        if (
            self.ref_stream_cache_enable
            and (not self.training)
            and self._cached_ref is not None
            and self._cached_ref.dim() == 3
            and int(self._cached_ref.size(0)) == int(ref.size(0))
            and int(self._cached_ref_len) > 0
        ):
            if ref_len < int(self._cached_ref_len):
                self._cached_ref = self._cached_ref[:, :, :ref_len]
                self._cached_ref_len = int(ref_len)
            use_ref_cache = (
                int(self._cached_ref_len) > 0
                and ref_len >= int(self._cached_ref_len)
                and self.ref_stream_block_frames > 0
            )

        if use_ref_cache:
            cached_len = int(self._cached_ref_len)
            new_frames = max(0, int(ref_len - cached_len))
            block = max(int(self.ref_stream_block_frames), int(new_frames))
            overlap = max(0, int(self.ref_stream_overlap_frames))
            start = max(0, ref_len - (block + overlap))
            ref_tail = ref[:, start:ref_len]
            ref_tail_feat = self.ref_encoder(ref_tail)
            if int(self._cached_ref.size(0)) != int(ref_tail_feat.size(0)):
                # Safety guard for variable batch sizes (e.g., last val/test batch).
                # Fall back to full tail features and refresh cache with current batch.
                ref = ref_tail_feat
                self._cached_ref = None
                self._cached_ref_len = 0
                use_ref_cache = False
            else:
                keep_from = max(0, ref_len - block)
                keep_tail = max(1, int(ref_tail_feat.size(2) - max(0, keep_from - start)))
                if keep_from > 0 and int(self._cached_ref.size(2)) >= keep_from:
                    ref = torch.cat([self._cached_ref[:, :, :keep_from], ref_tail_feat[:, :, -keep_tail:]], dim=2)
                else:
                    ref = ref_tail_feat[:, :, -keep_tail:]
                if int(ref.size(2)) > ref_len:
                    ref = ref[:, :, -ref_len:]
        else:
            ref = self.ref_encoder(ref)

        if self.ref_stream_cache_enable and (not self.training):
            self._cached_ref = ref.detach()
            self._cached_ref_len = int(ref.size(2))

        if st is not None:
            st["ref_encoder"] = float(st.get("ref_encoder", 0.0)) + time.perf_counter() - t0
            t1 = time.perf_counter()
        out = self.sep_network(mixture, ref)
        if st is not None:
            st["sep_network"] = float(st.get("sep_network", 0.0)) + time.perf_counter() - t1
        return out

    def clear_stream_cache(self):
        self._cached_ref = None
        self._cached_ref_len = 0
        self.sep_network.separator.clear_stream_cache()

    def trim_stream_cache(self, n_audio_drop: int = 0, n_ref_drop: int = 0) -> dict:
        """Drop leading cached frames to stay aligned with head-trimmed ring buffer."""
        a_drop = self.sep_network.separator.trim_stream_cache(int(n_audio_drop))
        r_drop = 0
        n_r = int(n_ref_drop)
        if n_r > 0 and self._cached_ref is not None:
            cur = int(self._cached_ref_len)
            n_r = min(n_r, cur)
            if n_r >= cur:
                self._cached_ref = None
                self._cached_ref_len = 0
                r_drop = cur
            elif n_r > 0:
                self._cached_ref = self._cached_ref[:, :, n_r:].contiguous()
                self._cached_ref_len = cur - n_r
                r_drop = n_r
        return {"audio_drop": int(a_drop), "ref_drop": int(r_drop)}


class AV_MossFormer2_TSE_16K(nn.Module):
    def __init__(self, args):
        super(AV_MossFormer2_TSE_16K, self).__init__()
        self.model = av_mossformer2(args)

    def forward(self, x):
        outputs = self.model(x)
        return outputs
