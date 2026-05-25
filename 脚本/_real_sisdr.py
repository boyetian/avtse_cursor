#!/usr/bin/env python3
"""Batch SI-SDR: pair target/{id}.wav with 测试结果/{id}_out.wav."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import soundfile as sf


def _to_mono(x: np.ndarray) -> np.ndarray:
    if x.ndim > 1:
        return x[:, 0]
    return x


def si_sdr(est: np.ndarray, ref: np.ndarray) -> float:
    est = est - np.mean(est)
    ref = ref - np.mean(ref)
    alpha = np.dot(est, ref) / (np.dot(ref, ref) + 1e-12)
    s_target = alpha * ref
    e_noise = est - s_target
    return float(
        10
        * np.log10(
            np.dot(s_target, s_target) / (np.dot(e_noise, e_noise) + 1e-12) + 1e-12
        )
    )


def _search_best_offset(
    est: np.ndarray, target: np.ndarray, coarse_step: int = 100, fine_radius: int = 100
) -> Tuple[float, int]:
    min_len = min(len(est), len(target))
    best_offset = 0
    best_val = -999.0

    def _eval_at(o: int) -> Optional[float]:
        if o >= 0:
            e = est[o:min_len]
            t = target[: min_len - o]
        else:
            e = est[: min_len + o]
            t = target[-o : min_len]
        if len(e) < 1000:
            return None
        return si_sdr(e.astype(np.float64), t.astype(np.float64))

    for offset in range(-8000, 8000, coarse_step):
        val = _eval_at(offset)
        if val is not None and val > best_val:
            best_val = val
            best_offset = offset

    for offset in range(best_offset - fine_radius, best_offset + fine_radius):
        val = _eval_at(offset)
        if val is not None and val > best_val:
            best_val = val
            best_offset = offset

    return best_val, best_offset


def eval_pair(
    target_path: str | Path,
    est_path: str | Path,
    do_align: bool = True,
) -> Tuple[float, int, float]:
    """Return (best_sisdr_db, offset_samples, no_align_sisdr_db)."""
    target = _to_mono(sf.read(str(target_path), dtype="float32")[0])
    est = _to_mono(sf.read(str(est_path), dtype="float32")[0])
    min_len = min(len(est), len(target))
    no_align = si_sdr(
        est[:min_len].astype(np.float64), target[:min_len].astype(np.float64)
    )
    if not do_align:
        return no_align, 0, no_align
    best_val, best_offset = _search_best_offset(est, target)
    return best_val, best_offset, no_align


def pair_target_out_dirs(
    target_dir: str | Path,
    out_dir: str | Path,
    out_suffix: str = "_out",
) -> List[Tuple[str, Path, Path]]:
    target_dir = Path(target_dir)
    out_dir = Path(out_dir)
    pairs: List[Tuple[str, Path, Path]] = []
    for ref in sorted(target_dir.glob("*.wav")):
        stem = ref.stem
        est = out_dir / f"{stem}{out_suffix}.wav"
        if est.is_file():
            pairs.append((stem, ref, est))
        else:
            print(f"[skip] no est for {ref.name} -> {est.name}")
    return pairs


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--target-dir", default="./测试用例/target")
    p.add_argument("--out-dir", default="./测试结果")
    p.add_argument("--out-suffix", default="_out", help="est filename = {stem}{suffix}.wav")
    p.add_argument("--single", action="store_true", help="evaluate one pair only")
    p.add_argument("--ref", default=None, help="reference wav (with --single)")
    p.add_argument("--est", default=None, help="estimate wav (with --single)")
    p.add_argument("--no-align", action="store_true", help="skip offset search")
    args = p.parse_args()

    if args.single:
        ref = args.ref or "测试用例/target/03.wav"
        est = args.est or "测试结果/03_out.wav"
        pairs = [("single", Path(ref), Path(est))]
    else:
        pairs = pair_target_out_dirs(args.target_dir, args.out_dir, args.out_suffix)

    if not pairs:
        print("No matched pairs found.")
        return

    scores: List[float] = []
    for stem, ref_path, est_path in pairs:
        best, offset, no_align = eval_pair(
            ref_path, est_path, do_align=not args.no_align
        )
        scores.append(best)
        off_ms = offset / 16000.0 * 1000.0
        print(
            f"{stem}: SI-SDR = {best:.3f} dB (offset={offset} samples, {off_ms:.1f} ms), "
            f"no-align = {no_align:.3f} dB"
        )

    mean = float(np.mean(scores))
    print(f"\nmean SI-SDR = {mean:.3f} dB ({len(scores)} pairs)")


if __name__ == "__main__":
    main()
