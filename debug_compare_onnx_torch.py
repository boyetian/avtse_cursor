"""Numerical comparison: torch vs ONNX split inference.

Feeds identical input to both backends and compares intermediate tensors
to pinpoint where the ONNX output diverges from torch.
"""

import os
import sys
import time

import numpy as np
import soundfile as sf
import torch
import yaml

from av_stream_inference import (
    AVStreamInference,
    _SplitOnnxModelWrapper,
    _load_model_weights,
    _dict_to_ns,
)
from networks import network_wrapper


def load_torch_model(config_path, checkpoint_dir, device):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    ns = _dict_to_ns(cfg)
    ns.device = device
    model = network_wrapper(ns).to(device)
    model.eval()
    ckpt = os.path.join(checkpoint_dir, "last_best_weights_only.pt")
    if not os.path.isfile(ckpt):
        ckpt = os.path.join(checkpoint_dir, "last_best_checkpoint.pt")
    _load_model_weights(model, ckpt)
    return model, ns


def rgb_to_gray_torch(rgb: torch.Tensor) -> torch.Tensor:
    """[B,T,H,W,3] -> [B,T,H,W] using training coefficients."""
    return (
        0.2989 * rgb[..., 0] + 0.5870 * rgb[..., 1] + 0.1140 * rgb[..., 2]
    )


def rgb_to_gray_np(rgb: np.ndarray) -> np.ndarray:
    """[B,T,H,W,3] -> [B,T,H,W] using training coefficients."""
    x = rgb.astype(np.float32, copy=False)
    return (
        0.2989 * x[..., 0] + 0.5870 * x[..., 1] + 0.1140 * x[..., 2]
    ).astype(np.float32, copy=False)


def main():
    config_path = "./checkpoints/AV_Mossformer/config.yaml"
    checkpoint_dir = "./checkpoints/AV_Mossformer"
    ref_onnx_path = "./checkpoints/AV_Mossformer/av_mossformer_ref_fixed.onnx"
    sep_onnx_path = "./checkpoints/AV_Mossformer/av_mossformer_sep_fixed.onnx"

    device = torch.device("cpu")
    print(f"Device: {device}")

    # 1. Load torch model
    print("[1/5] Loading torch model...")
    torch_model, ns = load_torch_model(config_path, checkpoint_dir, device)
    image_size = int(getattr(ns.network_audio, "image_size", 64))
    print(f"  image_size={image_size}")

    # 2. Load both ONNX split models (fixed + rknn)
    print("[2/5] Loading ONNX split models...")
    onnx_fixed = _SplitOnnxModelWrapper(
        ref_onnx_path, "./checkpoints/AV_Mossformer/av_mossformer_sep_fixed.onnx",
        image_size=image_size, num_threads=8,
    )
    onnx_rknn = _SplitOnnxModelWrapper(
        ref_onnx_path, "./checkpoints/AV_Mossformer/av_mossformer_sep_rknn.onnx",
        image_size=image_size, num_threads=8,
    )
    fixed_t_audio = onnx_fixed.fixed_t_audio
    fixed_t_ref = onnx_fixed.fixed_t_ref
    print(f"  fixed: t_audio={fixed_t_audio}, t_ref={fixed_t_ref}")
    print(f"  rknn:  t_audio={onnx_rknn.fixed_t_audio}, t_ref={onnx_rknn.fixed_t_ref}")

    # 3. Prepare test input (real audio + synthetic video to match fixed input size)
    print("[3/5] Preparing test input...")
    # Use real audio from test set
    audio_path = "./测试用例/音频/01.wav"
    if not os.path.exists(audio_path):
        # Fallback: find any wav
        audio_dir = "./测试用例/音频"
        wavs = sorted([f for f in os.listdir(audio_dir) if f.endswith(".wav")])
        audio_path = os.path.join(audio_dir, wavs[0]) if wavs else None

    if audio_path and os.path.exists(audio_path):
        wav, sr = sf.read(audio_path, dtype="float32")
        if wav.ndim == 2:
            wav = wav.mean(axis=1)
        print(f"  Loaded audio: {audio_path}, sr={sr}, len={len(wav)}")
    else:
        # Synthetic audio
        sr = 16000
        wav = np.random.randn(160000).astype(np.float32) * 0.01
        print(f"  Using synthetic audio, len={len(wav)}")

    # Ensure we have enough audio
    if len(wav) < fixed_t_audio:
        wav = np.pad(wav, (0, fixed_t_audio - len(wav)), mode="constant")

    # Take a window that matches the ONNX fixed input (skip first hop edge case)
    # Use the 2nd hop position: win_start=6400, win_end=16000
    win_start = 6400
    win_end = win_start + fixed_t_audio
    if win_end > len(wav):
        win_start = 0
        win_end = fixed_t_audio
    audio_chunk = wav[win_start:win_end].astype(np.float32)
    print(f"  Audio window: [{win_start}, {win_end}), shape={audio_chunk.shape}")

    # Create normalized video frames (simulate IncrementalVideoResampler output)
    # Use random frames with statistics matching real normalized data
    mean, std = 0.506362, 0.272887
    # Generate frames in [0,255] range, then normalize like real pipeline
    rng = np.random.RandomState(42)
    raw_frames = rng.randint(0, 256, size=(fixed_t_ref, image_size, image_size, 3), dtype=np.uint8)
    # Normalize: /255 then (x - mean) / std
    video_np = (raw_frames.astype(np.float32) / 255.0 - mean) / std
    video_np = video_np[np.newaxis, :]  # [1, T, H, W, 3]
    print(f"  Video shape: {video_np.shape}, range=[{video_np.min():.3f}, {video_np.max():.3f}]")

    # Torch tensor version
    audio_t = torch.from_numpy(audio_chunk).unsqueeze(0).to(device)  # [1, T]
    video_t = torch.from_numpy(video_np).to(device)  # [1, T, H, W, 3]

    # 4. Compare grayscale conversion
    print("\n[4/5] Comparing grayscale conversion...")
    gray_onnx = rgb_to_gray_np(video_np)  # [1, T, H, W]
    gray_torch = rgb_to_gray_torch(video_t).cpu().numpy()  # [1, T, H, W]
    gray_diff = np.abs(gray_onnx - gray_torch)
    print(f"  Grayscale max abs diff: {gray_diff.max():.6e}")
    print(f"  Grayscale mean abs diff: {gray_diff.mean():.6e}")
    if gray_diff.max() > 1e-5:
        print("  *** WARNING: Grayscale conversion differs!")

    # 5. Compare ref_encoder output (ref_feat)
    print("\n[5/5] Comparing ref_encoder and separator outputs...")

    # ONNX ref_encoder (same for both, using fixed model's ref encoder)
    t0 = time.perf_counter()
    gray_for_onnx = onnx_fixed._prepare_gray_ref(video_np)
    ref_feat_onnx = onnx_fixed.ref_sess.run(None, {"ref_gray": gray_for_onnx})[0]
    onnx_ref_time = time.perf_counter() - t0
    print(f"  ONNX ref_feat shape: {ref_feat_onnx.shape}, time={onnx_ref_time*1000:.2f}ms")

    # Torch ref_encoder (Visual_encoder)
    t0 = time.perf_counter()
    with torch.no_grad():
        gray_t = rgb_to_gray_torch(video_t)  # [1, T, H, W]
        ref_feat_torch = torch_model.model.ref_encoder(gray_t)  # [1, C, T]
    ref_feat_torch_np = ref_feat_torch.cpu().numpy()
    torch_ref_time = time.perf_counter() - t0
    print(f"  Torch ref_feat shape: {ref_feat_torch_np.shape}, time={torch_ref_time*1000:.2f}ms")

    ref_feat_diff = np.abs(ref_feat_onnx - ref_feat_torch_np)
    print(f"  ref_feat max abs diff: {ref_feat_diff.max():.6e}")
    print(f"  ref_feat mean abs diff: {ref_feat_diff.mean():.6e}")
    print(f"  ref_feat relative diff: {ref_feat_diff.max() / (np.abs(ref_feat_torch_np).max() + 1e-10):.6e}")

    # Compare separator outputs for BOTH ONNX models
    mix_for_onnx = audio_chunk.reshape(1, -1).astype(np.float32)

    with torch.no_grad():
        out_torch = torch_model.model.sep_network(audio_t, ref_feat_torch)
    out_torch_np = out_torch.squeeze().cpu().numpy()
    print(f"\n  Torch output shape: {out_torch_np.shape}, range=[{out_torch_np.min():.4f}, {out_torch_np.max():.4f}]")

    for name, onnx_model in [("sep_fixed", onnx_fixed), ("sep_rknn", onnx_rknn)]:
        print(f"\n--- {name} ---")
        t0 = time.perf_counter()
        out_onnx = onnx_model._run_sep(mix_for_onnx, ref_feat_torch_np)
        print(f"  time={ (time.perf_counter()-t0)*1000:.2f}ms")
        out_onnx_np = out_onnx.squeeze()
        print(f"  shape={out_onnx_np.shape}, range=[{out_onnx_np.min():.4f}, {out_onnx_np.max():.4f}]")

        out_diff = np.abs(out_onnx_np - out_torch_np)
        rel_diff = out_diff.max() / (np.abs(out_torch_np).max() + 1e-10)
        correlation = np.corrcoef(out_onnx_np, out_torch_np)[0, 1]
        scale_ratio = np.std(out_onnx_np) / (np.std(out_torch_np) + 1e-10)
        print(f"  vs torch: max abs diff={out_diff.max():.6e}, mean={out_diff.mean():.6e}, rel={rel_diff:.4f}")
        print(f"  correlation={correlation:.6f}, scale_ratio(std)={scale_ratio:.4f}")

        # Also compare vs each other
        if name == "sep_rknn":
            out_fixed_np = out_onnx_fixed_np
            cross_diff = np.abs(out_onnx_np - out_fixed_np)
            print(f"  vs fixed: max abs diff={cross_diff.max():.6e}, mean={cross_diff.mean():.6e}")
        else:
            out_onnx_fixed_np = out_onnx_np

    # Test: does ONNX separator use ref_feat at all?
    print("\n--- ref_feat ablation test ---")
    # Run ONNX separator with zero ref_feat
    zero_ref = np.zeros_like(ref_feat_torch_np)
    for name, onnx_model in [("sep_fixed", onnx_fixed), ("sep_rknn", onnx_rknn)]:
        out_zero_ref = onnx_model._run_sep(mix_for_onnx, zero_ref).squeeze()
        out_real_ref = onnx_model._run_sep(mix_for_onnx, ref_feat_torch_np).squeeze()
        diff = np.abs(out_zero_ref - out_real_ref).max()
        print(f"  {name}: max|output(real_ref) - output(zero_ref)| = {diff:.6e}")
        if diff < 1e-6:
            print(f"    *** WARNING: ref_feat has NO effect on output! ***")

    # Also test: ONNX with all-zero audio
    print("\n--- zero-input test ---")
    zero_mix = np.zeros((1, 9600), dtype=np.float32)
    for name, onnx_model in [("sep_fixed", onnx_fixed), ("sep_rknn", onnx_rknn)]:
        out_zero = onnx_model._run_sep(zero_mix, ref_feat_torch_np).squeeze()
        print(f"  {name}: output(zero_mix) max abs = {np.abs(out_zero).max():.6e}")
        if np.abs(out_zero).max() > 1e-6:
            print(f"    *** WARNING: non-zero output for zero input! ***")

    # Summary
    print("\n=== SUMMARY ===")
    print(f"  Grayscale: {'PASS' if gray_diff.max() <= 1e-5 else 'FAIL'} (max diff={gray_diff.max():.2e})")
    print(f"  ref_feat:  {'PASS' if ref_feat_diff.max() <= 0.01 else 'FAIL'} (max diff={ref_feat_diff.max():.2e})")

    for name, out_onnx_np in [("fixed", out_onnx_fixed_np), ("rknn", onnx_rknn._run_sep(mix_for_onnx, ref_feat_torch_np).squeeze())]:
        diff = np.abs(out_onnx_np - out_torch_np)
        ok = diff.max() <= 0.01
        print(f"  sep_{name}: {'PASS' if ok else 'FAIL'} (max abs diff={diff.max():.4f}, rel={diff.max()/(np.abs(out_torch_np).max()+1e-10):.4f})")


if __name__ == "__main__":
    main()
