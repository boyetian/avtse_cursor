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


def convert_pcm_dir_to_wav_dir(
    pcm_dir: str,
    out_dir: str,
    *,
    sr: int = 16000,
    dtype: str = "int16",
    mode: str = "clean",
) -> list[str]:
    """将文件夹下所有 PCM 混成单通道 WAV，返回已转换文件的路径列表。"""
    pcm_files = sorted([f for f in os.listdir(pcm_dir) if f.lower().endswith(".pcm")])
    if not pcm_files:
        raise FileNotFoundError(f"目录 {pcm_dir} 中未找到 .pcm 文件")

    os.makedirs(out_dir, exist_ok=True)
    converted = []

    for i, pcm_file in enumerate(pcm_files):
        pcm_path = os.path.join(pcm_dir, pcm_file)
        stem = os.path.splitext(pcm_file)[0]
        out_path = os.path.join(out_dir, f"{stem}_{mode}.wav")

        convert_pcm_to_wav(pcm_path, out_path, sr=sr, dtype=dtype, mode=mode)
        converted.append(out_path)
        print(f"[{i + 1}/{len(pcm_files)}] 已转换: {pcm_file} -> {stem}_{mode}.wav")

    return converted


def main() -> None:
    p = argparse.ArgumentParser(description="8ch PCM -> mono wav")
    p.add_argument("--pcm", default=None, help="单个 8 通道 PCM 文件路径")
    p.add_argument("--pcm_dir", default=None, help="PCM 文件目录（批量模式）")
    p.add_argument("--out", default=None, help="单文件模式输出路径")
    p.add_argument("--out_dir", default=None, help="批量模式输出目录 (默认与 PCM_DIR 同目录)")
    p.add_argument("--sr", type=int, default=16000)
    p.add_argument("--dtype", default="int16")
    p.add_argument("--mode", default="clean", choices=["clean", "noisy"])
    args = p.parse_args()

    # 批量模式
    if args.pcm_dir:
        out_dir = args.out_dir or args.pcm_dir
        converted = convert_pcm_dir_to_wav_dir(
            args.pcm_dir, out_dir, sr=args.sr, dtype=args.dtype, mode=args.mode
        )
        print(f"\n[完成] 共转换 {len(converted)} 个文件 -> {out_dir}")
        return

    # 单文件模式
    if not args.pcm:
        p.error("请指定 --pcm（单文件）或 --pcm_dir（批量）")
    out = args.out
    if out is None:
        stem = os.path.splitext(os.path.basename(args.pcm))[0]
        out = os.path.join(os.path.dirname(args.pcm) or ".", f"{stem}_{args.mode}.wav")
    convert_pcm_to_wav(args.pcm, out, sr=args.sr, dtype=args.dtype, mode=args.mode)
    info = sf.info(out)
    print(f"out: {out}")
    print(f"sr={info.samplerate} Hz, duration={info.duration:.3f}s")


if __name__ == "__main__":
    main()
