"""8 通道 PCM -> 单通道 WAV（供 main.py 与命令行使用）。"""
from __future__ import annotations

import argparse
import os
import sys

import soundfile as sf

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PREPARE_DIR = os.path.join(_SCRIPT_DIR, "读文件脚本")
if _PREPARE_DIR not in sys.path:
    sys.path.insert(0, _PREPARE_DIR)

from prepare_chunk_input import mix_channels, read_8ch_pcm  # noqa: E402


def convert_pcm_to_wav(
    pcm_path: str,
    out_path: str,
    *,
    sr: int = 16000,
    dtype: str = "int16",
    mode: str = "clean",
) -> str:
    """将 8ch 交错 PCM 混成单通道并写入 WAV，返回 out_path。"""
    pcm_8ch, _ = read_8ch_pcm(pcm_path, int(sr), dtype)
    mono = mix_channels(pcm_8ch, mode)
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    sf.write(out_path, mono, int(sr), subtype="PCM_16")
    return out_path


def main() -> None:
    p = argparse.ArgumentParser(description="8ch PCM -> mono wav")
    p.add_argument("--pcm", required=True)
    p.add_argument("--out", default=None, help="default: <stem>_<mode>.wav beside pcm")
    p.add_argument("--sr", type=int, default=16000)
    p.add_argument("--dtype", default="int16")
    p.add_argument("--mode", default="clean", choices=["clean", "noisy"])
    args = p.parse_args()
    out = args.out
    if out is None:
        stem = os.path.splitext(os.path.basename(args.pcm))[0]
        out = os.path.join(os.path.dirname(args.pcm) or ".", f"{stem}_{args.mode}.wav")
    convert_pcm_to_wav(
        args.pcm, out, sr=args.sr, dtype=args.dtype, mode=args.mode
    )
    info = sf.info(out)
    print(f"out: {out}")
    print(f"sr={info.samplerate} Hz, duration={info.duration:.3f}s")


if __name__ == "__main__":
    main()
