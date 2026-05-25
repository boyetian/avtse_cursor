#!/usr/bin/env python3
"""Build RKNN INT8 calibration set for av_mossformer_sep (mixture + ref_feat).

Pipeline:
  1) Sample source pairs (mixture wav + full-frame ref mp4) from 2mix CSV:
       cd ../target_speaker_extraction_online
       python data/sample_2mix_calib_pairs.py --num 200 \\
         --out-dir ../AV_TSE/checkpoints/AV_Mossformer/rknn_calib_src_200
  2) Build calibration npy + dataset.txt (this script; MediaPipe lip crop like main.py):
       python scripts/build_rknn_sep_calib.py \\
         --pairs-dir checkpoints/AV_Mossformer/rknn_calib_src_200 \\
         --max-samples 200 \\
         --preview-dir checkpoints/AV_Mossformer/rknn_calib_preview \\
         --preview-max 20
  3) Convert to RKNN INT8 (RKNN-Toolkit2 env):
       python convert_av_mossformer_rknn.py --dtype i8 \\
         --dataset checkpoints/AV_Mossformer/rknn_calib_sep/dataset.txt
"""

from __future__ import annotations

import argparse
import glob
import math
import os
import sys
from dataclasses import dataclass
from typing import Optional

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# Match av_stream_inference.IncrementalVideoResampler (mossformer path)
MOSSFORMER_MEAN = 0.506362
MOSSFORMER_STD = 0.272877


@dataclass
class VideoPreprocessConfig:
    image_size: int = 96
    ref_sr: float = 30.0
    use_mediapipe_lip: bool = True
    face_detector_model: str = "detector.tflite"
    mediapipe_lip_crop_scale: float = 0.8
    mediapipe_lip_crop_min_px: int = 48
    mediapipe_lip_crop_max_px: int = 2048
    detect_every_n: int = 5
    face_scale: float = 0.8
    box_smooth_alpha: float = 0.85
    face_target_policy: str = "center_largest"
    face_target_lock: bool = True
    face_target_lock_min_iou: float = 0.15


def _compute_stream_window_lengths(
    audio_sr: int = 16000,
    ref_sr: float = 30.0,
    context_ms: float = 100.0,
    infer_chunk_ms: float = 200.0,
    lookahead_ms: float = 0.0,
) -> tuple[int, int]:
    context_samples = max(0, int(round(float(audio_sr) * (float(context_ms) / 1000.0))))
    hop_samples = max(1, int(round(float(audio_sr) * (float(infer_chunk_ms) / 1000.0))))
    lookahead_samples = max(0, int(round(float(audio_sr) * (float(lookahead_ms) / 1000.0))))
    t_audio = max(256, context_samples + hop_samples + lookahead_samples)
    t_ref = max(2, int(round(float(t_audio) / float(audio_sr) * float(ref_sr))))
    return int(t_audio), int(t_ref)


def _resample_audio(mono: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out or mono.size == 0:
        return mono.astype(np.float32, copy=False)
    try:
        import torch
        import torchaudio

        t = torch.from_numpy(mono.astype(np.float32))
        return torchaudio.functional.resample(t, sr_in, sr_out).numpy()
    except ImportError:
        import librosa

        return librosa.resample(mono, orig_sr=sr_in, target_sr=sr_out).astype(np.float32)


def _load_mono_wav(path: str, audio_sr: int) -> np.ndarray:
    import soundfile as sf

    wav, sr = sf.read(path, dtype="float32", always_2d=True)
    mono = wav.mean(axis=1) if wav.shape[1] > 1 else wav[:, 0]
    return _resample_audio(mono, int(sr), audio_sr)


def _load_bgr_frames_uint8(mp4_path: str, ref_sr: float) -> tuple[list[np.ndarray], float]:
    import cv2

    cap = cv2.VideoCapture(mp4_path)
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if not math.isfinite(fps) or fps <= 1e-3:
        fps = ref_sr
    frames: list[np.ndarray] = []
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        frames.append(bgr)
    cap.release()

    if fps != ref_sr and len(frames) > 0:
        duration_s = len(frames) / fps
        tgt_len = max(1, int(round(duration_s * ref_sr)))
        resampled: list[np.ndarray] = []
        for ti in range(tgt_len):
            si = int(np.clip(round((ti / ref_sr) * fps), 0, len(frames) - 1))
            resampled.append(frames[si])
        frames = resampled
    return frames, fps


def _make_mediapipe_tracker(vcfg: VideoPreprocessConfig):
    from face_mediapipe_tracker import FaceMediaPipeStreamTracker

    model_path = vcfg.face_detector_model
    if not os.path.isabs(model_path):
        model_path = os.path.join(_ROOT, model_path)
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"MediaPipe model not found: {model_path}")
    return FaceMediaPipeStreamTracker(
        crop_size=vcfg.image_size,
        face_scale=vcfg.face_scale,
        detect_every_n=vcfg.detect_every_n,
        box_smooth_alpha=vcfg.box_smooth_alpha,
        model_path=model_path,
        use_lip_center_crop=vcfg.use_mediapipe_lip,
        lip_crop_scale=vcfg.mediapipe_lip_crop_scale,
        lip_crop_min_px=vcfg.mediapipe_lip_crop_min_px,
        lip_crop_max_px=vcfg.mediapipe_lip_crop_max_px,
        target_policy=vcfg.face_target_policy,
        target_lock=vcfg.face_target_lock,
        target_lock_min_iou=vcfg.face_target_lock_min_iou,
    )


def _lip_crops_from_mp4(
    mp4_path: str, vcfg: VideoPreprocessConfig
) -> tuple[list[np.ndarray], list[bool], float]:
    """Per-frame lip RGB crops (float32 ~0-255) + face-detected flags, aligned to ref_sr."""
    bgr_frames, fps = _load_bgr_frames_uint8(mp4_path, vcfg.ref_sr)
    tracker = _make_mediapipe_tracker(vcfg)
    lip_rgb: list[np.ndarray] = []
    valid: list[bool] = []
    for bgr in bgr_frames:
        face_rgb, _ = tracker.process_bgr(bgr)
        lip_rgb.append(face_rgb)
        valid.append(tracker.last_box is not None)
    return lip_rgb, valid, fps


def _load_rgb_frames_fullframe(
    mp4_path: str, image_size: int, ref_sr: float
) -> tuple[list[np.ndarray], float]:
    """Legacy: whole-frame resize (no MediaPipe). RGB float in [0,1]."""
    import cv2

    cap = cv2.VideoCapture(mp4_path)
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if not math.isfinite(fps) or fps <= 1e-3:
        fps = ref_sr
    frames: list[np.ndarray] = []
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        rgb = cv2.resize(rgb, (image_size, image_size), interpolation=cv2.INTER_AREA)
        frames.append(rgb)
    cap.release()

    if fps != ref_sr and len(frames) > 0:
        duration_s = len(frames) / fps
        tgt_len = max(1, int(round(duration_s * ref_sr)))
        resampled = []
        for ti in range(tgt_len):
            si = int(np.clip(round((ti / ref_sr) * fps), 0, len(frames) - 1))
            resampled.append(frames[si])
        frames = resampled
    return frames, fps


def _normalize_mossformer_rgb(frames_255: list[np.ndarray]) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    for fr in frames_255:
        x = fr.astype(np.float32, copy=False) / 255.0
        x = (x - MOSSFORMER_MEAN) / MOSSFORMER_STD
        out.append(x.astype(np.float32, copy=False))
    return out


def _rgb_stack_to_gray(ref_bt_hwc: np.ndarray) -> np.ndarray:
    """Match networks.network_wrapper / _SplitOnnxModelWrapper ([B,T,H,W,3] -> [B,T,H,W])."""
    x = ref_bt_hwc.astype(np.float32)
    if x.ndim != 5 or x.shape[-1] != 3:
        raise ValueError(f"expected [B,T,H,W,3], got {x.shape}")
    gray = 0.2989 * x[..., 0] + 0.5870 * x[..., 1] + 0.1140 * x[..., 2]
    return gray.astype(np.float32)


def _infer_shapes_from_onnx(sep_onnx: str) -> tuple[int, int, int]:
    import onnx

    shapes: dict[str, list[int]] = {}
    m = onnx.load(sep_onnx)
    for inp in m.graph.input:
        dims = [int(d.dim_value) for d in inp.type.tensor_type.shape.dim if d.dim_value > 0]
        shapes[inp.name] = dims
    audio_len = shapes.get("mixture", [1, 0])[-1]
    rf = shapes.get("ref_feat", [1, 96, 0])
    ref_ch = rf[1] if len(rf) >= 2 else 96
    ref_frames = rf[2] if len(rf) >= 3 else 0
    if audio_len <= 0 or ref_frames <= 0:
        raise ValueError(f"cannot read fixed shapes from {sep_onnx}: {shapes}")
    return int(audio_len), int(ref_ch), int(ref_frames)


def _write_video_mp4(path: str, frames: list[np.ndarray], fps: float) -> bool:
    import cv2

    if not frames:
        return False
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, float(fps), (int(w), int(h)))
    if not writer.isOpened():
        return False
    for fr in frames:
        if fr.ndim == 2:
            u8 = np.clip(fr * 255.0, 0, 255).astype(np.uint8)
            bgr = cv2.cvtColor(u8, cv2.COLOR_GRAY2BGR)
        else:
            u8 = np.clip(fr, 0, 255).astype(np.uint8)
            if u8.shape[-1] == 3:
                bgr = cv2.cvtColor(u8, cv2.COLOR_RGB2BGR)
            else:
                bgr = u8
        if bgr.shape[0] != h or bgr.shape[1] != w:
            bgr = cv2.resize(bgr, (w, h), interpolation=cv2.INTER_AREA)
        writer.write(bgr)
    writer.release()
    return os.path.isfile(path)


def _write_preview_clip(
    preview_dir: str,
    idx: int,
    lip_rgb_255: np.ndarray,
    gray_bt_hw: np.ndarray,
    mix_1d: np.ndarray,
    ref_sr: float,
    audio_sr: int,
) -> None:
    import soundfile as sf

    os.makedirs(preview_dir, exist_ok=True)
    lip_path = os.path.join(preview_dir, f"calib_{idx:04d}_lip_rgb.mp4")
    gray_path = os.path.join(preview_dir, f"calib_{idx:04d}_gray.mp4")
    wav_path = os.path.join(preview_dir, f"calib_{idx:04d}_mix.wav")

    lip_frames = [lip_rgb_255[t] for t in range(lip_rgb_255.shape[0])]
    g = gray_bt_hw[0].astype(np.float32)
    lo, hi = float(np.percentile(g, 2)), float(np.percentile(g, 98))
    span = max(hi - lo, 1e-6)
    gray_frames = [np.clip((g[t] - lo) / span, 0.0, 1.0) for t in range(g.shape[0])]

    if not _write_video_mp4(lip_path, lip_frames, ref_sr):
        print(f"[calib] WARN: failed to write {lip_path}", flush=True)
    if not _write_video_mp4(gray_path, gray_frames, ref_sr):
        print(f"[calib] WARN: failed to write {gray_path}", flush=True)
    sf.write(wav_path, mix_1d.astype(np.float32), int(audio_sr))


def _write_sample(
    out_dir: str,
    idx: int,
    mix: np.ndarray,
    ref_feat: np.ndarray,
    lines: list[str],
    name_prefix: str = "mix",
    feat_prefix: str = "ref_feat",
) -> None:
    mix_path = os.path.join(out_dir, f"{name_prefix}_{idx:04d}.npy")
    feat_path = os.path.join(out_dir, f"{feat_prefix}_{idx:04d}.npy")
    np.save(mix_path, mix.astype(np.float32))
    np.save(feat_path, ref_feat.astype(np.float32))
    lines.append(f"{os.path.abspath(mix_path)} {os.path.abspath(feat_path)}")


def build_from_av(
    wav_path: str,
    mp4_path: str,
    ref_sess,
    out_dir: str,
    n_calib: int,
    audio_len: int,
    ref_frames: int,
    vcfg: VideoPreprocessConfig,
    audio_sr: int,
    start_idx: int = 0,
    preview_dir: str = "",
    preview_budget: int = 0,
) -> tuple[int, list[str], int]:
    wav = _load_mono_wav(wav_path, audio_sr)

    if vcfg.use_mediapipe_lip:
        lip_list, valid_list, fps = _lip_crops_from_mp4(mp4_path, vcfg)
        norm_list = _normalize_mossformer_rgb(lip_list)
        mode = "mediapipe_lip"
    else:
        norm_list, fps = _load_rgb_frames_fullframe(mp4_path, vcfg.image_size, vcfg.ref_sr)
        lip_list = [fr * 255.0 for fr in norm_list]
        valid_list = [True] * len(norm_list)
        mode = "fullframe_resize"

    n_audio = wav.shape[0] // audio_len
    n_video = len(norm_list) // ref_frames
    n_chunks = min(n_audio, n_video, n_calib)
    if n_chunks <= 0:
        raise RuntimeError(
            f"not enough aligned data: audio_chunks={n_audio}, video_chunks={n_video}, "
            f"need audio_len={audio_len}, ref_frames={ref_frames}"
        )
    print(
        f"[calib] wav={os.path.basename(wav_path)} mp4={os.path.basename(mp4_path)} "
        f"video={mode} fps={fps:.2f}->{vcfg.ref_sr} chunks={n_chunks} "
        f"(audio {n_audio}, video {n_video})"
    )

    lines: list[str] = []
    previews_left = int(preview_budget)
    written = 0
    for i in range(n_chunks):
        v0 = i * ref_frames
        window_valid = valid_list[v0 : v0 + ref_frames]
        if vcfg.use_mediapipe_lip and not all(window_valid):
            continue

        a0 = i * audio_len
        mix = wav[a0 : a0 + audio_len][np.newaxis, :].astype(np.float32)

        norm_window = norm_list[v0 : v0 + ref_frames]
        ref_rgb_norm = np.stack(norm_window, axis=0)[np.newaxis, ...]
        gray = _rgb_stack_to_gray(ref_rgb_norm)
        ref_feat = ref_sess.run(None, {"ref_gray": gray})[0]

        if mix.shape != (1, audio_len):
            raise ValueError(f"mix shape {mix.shape} != (1, {audio_len})")
        if ref_feat.shape[0] != 1 or ref_feat.shape[2] != ref_frames:
            raise ValueError(f"ref_feat shape {ref_feat.shape}, expected T={ref_frames}")

        out_idx = start_idx + written
        _write_sample(out_dir, out_idx, mix, ref_feat, lines)

        if preview_dir and previews_left > 0:
            lip_window = np.stack(lip_list[v0 : v0 + ref_frames], axis=0)
            _write_preview_clip(
                preview_dir,
                out_idx,
                lip_window,
                gray,
                mix[0],
                vcfg.ref_sr,
                audio_sr,
            )
            previews_left -= 1
        written += 1

    return written, lines, previews_left


def _discover_pairs(pairs_dir: str, prefix: str) -> list[tuple[str, str]]:
    pattern = os.path.join(pairs_dir, f"{prefix}_*.wav")
    wavs = sorted(glob.glob(pattern))
    pairs: list[tuple[str, str]] = []
    for wav_path in wavs:
        base, _ = os.path.splitext(wav_path)
        mp4_path = base + ".mp4"
        if os.path.isfile(mp4_path):
            pairs.append((wav_path, mp4_path))
    return pairs


def build_from_pairs_dir(
    pairs_dir: str,
    pairs_prefix: str,
    ref_sess,
    out_dir: str,
    max_samples: int,
    audio_len: int,
    ref_frames: int,
    vcfg: VideoPreprocessConfig,
    audio_sr: int,
    preview_dir: str = "",
    preview_max: int = 0,
) -> tuple[int, list[str]]:
    pairs = _discover_pairs(pairs_dir, pairs_prefix)
    if not pairs:
        raise RuntimeError(f"no {pairs_prefix}_*.wav + .mp4 under {pairs_dir}")
    print(f"[calib] pairs-dir: {len(pairs)} files under {pairs_dir}")

    lines: list[str] = []
    written = 0
    previews_left = preview_max if preview_dir else 0
    per_pair_cap = (
        max(1, (max_samples + len(pairs) - 1) // len(pairs)) if max_samples > 0 else 10**9
    )
    skipped = 0
    for wav_path, mp4_path in pairs:
        if max_samples > 0 and written >= max_samples:
            break
        budget = per_pair_cap
        if max_samples > 0:
            budget = min(budget, max_samples - written)
        try:
            n, chunk_lines, previews_left = build_from_av(
                wav_path,
                mp4_path,
                ref_sess,
                out_dir,
                budget,
                audio_len,
                ref_frames,
                vcfg,
                audio_sr,
                start_idx=written,
                preview_dir=preview_dir,
                preview_budget=previews_left,
            )
        except (RuntimeError, FileNotFoundError) as e:
            print(f"[calib] skip {os.path.basename(wav_path)}: {e}", flush=True)
            skipped += 1
            continue
        if n <= 0:
            skipped += 1
            continue
        lines.extend(chunk_lines)
        written += n
    if skipped:
        print(f"[calib] skipped {skipped} pair(s) (no valid chunks or IO error)", flush=True)
    return written, lines


def build_random(
    ref_sess,
    out_dir: str,
    n_calib: int,
    audio_len: int,
    ref_frames: int,
    image_size: int,
) -> tuple[int, list[str]]:
    print(f"[calib] random fallback: {n_calib} samples (smoke test only)")
    lines: list[str] = []
    for i in range(n_calib):
        mix = np.random.randn(1, audio_len).astype(np.float32) * 0.05
        ref_rgb = np.random.rand(1, ref_frames, image_size, image_size, 3).astype(np.float32)
        gray = _rgb_stack_to_gray(ref_rgb)
        ref_feat = ref_sess.run(None, {"ref_gray": gray})[0]
        _write_sample(out_dir, i, mix, ref_feat, lines)
    return n_calib, lines


def main() -> int:
    parser = argparse.ArgumentParser(description="RKNN sep INT8 calibration npy + dataset.txt")
    parser.add_argument("--wav", type=str, default="", help="Calibration wav (mono or multi-channel)")
    parser.add_argument("--mp4", type=str, default="", help="Calibration mp4 (aligned with wav)")
    parser.add_argument(
        "--ref_onnx",
        type=str,
        default="checkpoints/AV_Mossformer/av_mossformer_ref_fixed.onnx",
    )
    parser.add_argument(
        "--sep_onnx",
        type=str,
        default="checkpoints/AV_Mossformer/av_mossformer_sep_rknn.onnx",
        help="Read mixture/ref_feat shapes from sep ONNX",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="checkpoints/AV_Mossformer/rknn_calib_sep",
    )
    parser.add_argument("--n_calib", type=int, default=200)
    parser.add_argument("--audio_sr", type=int, default=16000)
    parser.add_argument("--ref_sr", type=float, default=30.0)
    parser.add_argument("--context_ms", type=float, default=100.0)
    parser.add_argument("--infer_chunk_ms", type=float, default=200.0)
    parser.add_argument("--image_size", type=int, default=96)
    parser.add_argument("--audio_len", type=int, default=0, help="0=from sep_onnx or stream window")
    parser.add_argument("--ref_frames", type=int, default=0, help="0=from sep_onnx or stream window")
    parser.add_argument(
        "--random_only",
        action="store_true",
        help="Skip wav/mp4; generate random mixture+video (smoke test only)",
    )
    parser.add_argument(
        "--pairs-dir",
        type=str,
        default="",
        help="Directory with pair_XXXX.wav and pair_XXXX.mp4 (from sample_2mix_calib_pairs.py)",
    )
    parser.add_argument(
        "--pairs-prefix",
        type=str,
        default="pair",
        help="Filename prefix for --pairs-dir (default: pair -> pair_0000.wav)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Total calibration npy count for --pairs-dir (0 = use --n_calib)",
    )
    parser.add_argument(
        "--preview-dir",
        type=str,
        default="",
        help="Write calib_*_lip_rgb.mp4, calib_*_gray.mp4, calib_*_mix.wav for inspection",
    )
    parser.add_argument(
        "--preview-max",
        type=int,
        default=20,
        help="Max preview clips per run (0 = unlimited). Ignored if --preview-dir empty.",
    )
    parser.add_argument(
        "--face-detector-model",
        type=str,
        default="detector.tflite",
        help="MediaPipe face model (relative to AV_TSE root)",
    )
    parser.add_argument(
        "--mediapipe-lip-crop",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Lip-centered crop like main.py (default: on)",
    )
    parser.add_argument("--mediapipe-lip-crop-scale", type=float, default=0.8)
    parser.add_argument("--mediapipe-lip-crop-min-px", type=int, default=48)
    parser.add_argument("--mediapipe-lip-crop-max-px", type=int, default=2048)
    parser.add_argument("--detect-every-n", type=int, default=5)
    parser.add_argument("--face-scale", type=float, default=0.8)
    parser.add_argument("--box-smooth-alpha", type=float, default=0.85)
    parser.add_argument(
        "--face-target-policy",
        type=str,
        default="center_largest",
        choices=["largest", "center", "center_largest", "center_largest_lock"],
    )
    parser.add_argument(
        "--face-target-lock",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--face-target-lock-min-iou", type=float, default=0.15)
    args = parser.parse_args()

    mode_count = sum(
        [
            bool(args.random_only),
            bool(args.wav and args.mp4),
            bool(args.pairs_dir),
        ]
    )
    if mode_count != 1:
        print(
            "Choose exactly one: --random_only, (--wav and --mp4), or --pairs-dir",
            file=sys.stderr,
        )
        return 1

    if args.audio_len > 0 and args.ref_frames > 0:
        audio_len, ref_ch, ref_frames = int(args.audio_len), 96, int(args.ref_frames)
    elif os.path.isfile(args.sep_onnx):
        try:
            audio_len, ref_ch, ref_frames = _infer_shapes_from_onnx(
                os.path.abspath(args.sep_onnx)
            )
        except ImportError:
            print("[calib] onnx not installed; using stream window lengths", flush=True)
            audio_len, ref_frames = _compute_stream_window_lengths(
                audio_sr=args.audio_sr,
                ref_sr=args.ref_sr,
                context_ms=args.context_ms,
                infer_chunk_ms=args.infer_chunk_ms,
            )
            ref_ch = 96
    else:
        audio_len, ref_frames = _compute_stream_window_lengths(
            audio_sr=args.audio_sr,
            ref_sr=args.ref_sr,
            context_ms=args.context_ms,
            infer_chunk_ms=args.infer_chunk_ms,
        )
        ref_ch = 96

    vcfg = VideoPreprocessConfig(
        image_size=int(args.image_size),
        ref_sr=float(args.ref_sr),
        use_mediapipe_lip=bool(args.mediapipe_lip_crop),
        face_detector_model=str(args.face_detector_model),
        mediapipe_lip_crop_scale=float(args.mediapipe_lip_crop_scale),
        mediapipe_lip_crop_min_px=int(args.mediapipe_lip_crop_min_px),
        mediapipe_lip_crop_max_px=int(args.mediapipe_lip_crop_max_px),
        detect_every_n=int(args.detect_every_n),
        face_scale=float(args.face_scale),
        box_smooth_alpha=float(args.box_smooth_alpha),
        face_target_policy=str(args.face_target_policy),
        face_target_lock=bool(args.face_target_lock),
        face_target_lock_min_iou=float(args.face_target_lock_min_iou),
    )

    ref_onnx = os.path.abspath(args.ref_onnx)
    if not os.path.isfile(ref_onnx):
        print(f"ref ONNX not found: {ref_onnx}", file=sys.stderr)
        return 1

    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    preview_dir = os.path.abspath(args.preview_dir) if args.preview_dir else ""
    preview_max = int(args.preview_max) if preview_dir else 0
    if preview_dir and preview_max == 0:
        preview_max = 10**9

    import onnxruntime as ort

    ref_sess = ort.InferenceSession(ref_onnx, providers=["CPUExecutionProvider"])
    print(
        f"[calib] shapes mixture=[1,{audio_len}] ref_feat=[1,{ref_ch},{ref_frames}] "
        f"image_size={vcfg.image_size} video={'mediapipe_lip' if vcfg.use_mediapipe_lip else 'fullframe'}"
    )
    if preview_dir:
        print(f"[calib] preview-dir={preview_dir} preview-max={preview_max}")

    if args.random_only:
        n_written, lines = build_random(
            ref_sess, out_dir, args.n_calib, audio_len, ref_frames, vcfg.image_size
        )
    elif args.pairs_dir:
        pairs_dir = os.path.abspath(args.pairs_dir)
        if not os.path.isdir(pairs_dir):
            print(f"pairs-dir not found: {pairs_dir}", file=sys.stderr)
            return 1
        max_samples = int(args.max_samples) if args.max_samples > 0 else int(args.n_calib)
        n_written, lines = build_from_pairs_dir(
            pairs_dir,
            args.pairs_prefix,
            ref_sess,
            out_dir,
            max_samples,
            audio_len,
            ref_frames,
            vcfg,
            args.audio_sr,
            preview_dir=preview_dir,
            preview_max=preview_max,
        )
        if n_written <= 0:
            print("ERROR: no calibration samples from pairs-dir", file=sys.stderr)
            return 1
    else:
        wav_path = os.path.abspath(args.wav)
        mp4_path = os.path.abspath(args.mp4)
        if not os.path.isfile(wav_path) or not os.path.isfile(mp4_path):
            print("wav/mp4 not found", file=sys.stderr)
            return 1
        n_written, lines, _ = build_from_av(
            wav_path,
            mp4_path,
            ref_sess,
            out_dir,
            args.n_calib,
            audio_len,
            ref_frames,
            vcfg,
            args.audio_sr,
            preview_dir=preview_dir,
            preview_budget=preview_max,
        )

    dataset_txt = os.path.join(out_dir, "dataset.txt")
    with open(dataset_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + ("\n" if lines else ""))
    sep_base = os.path.splitext(os.path.abspath(args.sep_onnx))[0]
    print(f"[calib] wrote {n_written} samples -> {out_dir}")
    print(f"[calib] dataset: {dataset_txt}")
    if preview_dir:
        print(f"[calib] previews -> {preview_dir}")
    print(
        "\nNext:\n"
        f"  python convert_av_mossformer_rknn.py \\\n"
        f"    --model {args.sep_onnx} \\\n"
        f"    --dtype i8 \\\n"
        f"    --dataset {dataset_txt} \\\n"
        f"    --output_path {sep_base}_i8.rknn"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
