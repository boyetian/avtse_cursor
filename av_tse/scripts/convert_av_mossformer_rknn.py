#!/usr/bin/env python3
"""Convert AV-Mossformer ONNX to RKNN with fixed mixture/ref shapes for RK3588."""

from __future__ import annotations

import argparse
import math
import os
import sys

def _find_asr_scripts_dir() -> str:
    """Resolve asr_frontend/scripts whether this file lives under AV_TSE/ or av_tse/scripts/."""
    here = os.path.dirname(os.path.abspath(__file__))
    for rel in (
        os.path.join("asr_frontend", "scripts"),              # AV_TSE/convert_*.py
        os.path.join("..", "..", "asr_frontend", "scripts"),  # av_tse/scripts/convert_*.py
    ):
        cand = os.path.normpath(os.path.join(here, rel))
        if os.path.isfile(os.path.join(cand, "rknn_mem_guard.py")):
            return cand
    raise ImportError(
        f"rknn_mem_guard.py not found under {here}/../asr_frontend/scripts "
        f"(place asr_frontend next to av_tse under AV_TSE/)"
    )


_ASR_SCRIPTS = _find_asr_scripts_dir()
if _ASR_SCRIPTS not in sys.path:
    sys.path.insert(0, _ASR_SCRIPTS)

from rknn_mem_guard import require_avail_gb  # noqa: E402

DEFAULT_ONNX = "../../../av_tse/checkpoints/AV_Mossformer/av_mossformer2.onnx"
DEFAULT_AUDIO_SR = 16000
DEFAULT_REF_SR = 30.0
DEFAULT_CONTEXT_MS = 100.0
DEFAULT_INFER_CHUNK_MS = 500.0
DEFAULT_IMAGE_SIZE = 96


def default_audio_len(
    audio_sr: int,
    infer_chunk_ms: float,
    context_ms: float,
    lookahead_ms: float = 0.0,
) -> int:
    hop = int(round(audio_sr * infer_chunk_ms / 1000.0))
    context = int(round(audio_sr * context_ms / 1000.0))
    lookahead = int(round(audio_sr * lookahead_ms / 1000.0))
    return hop + context + lookahead


def default_ref_frames(audio_len: int, audio_sr: int, ref_sr: float, margin: int = 2) -> int:
    frames = int(math.ceil(float(audio_len) / float(audio_sr) * ref_sr))
    return max(2, frames + margin)


def main() -> int:
    parser = argparse.ArgumentParser(description="AV-Mossformer ONNX -> RKNN")
    parser.add_argument("--model", type=str, default=DEFAULT_ONNX, help="ONNX model path")
    parser.add_argument("--output_path", type=str, default="", help="Output .rknn path")
    parser.add_argument("--target", type=str, default="rk3588", help="RKNN target platform")
    parser.add_argument("--dtype", type=str, default="fp", choices=("fp", "i8"))
    parser.add_argument("--audio_sr", type=int, default=DEFAULT_AUDIO_SR)
    parser.add_argument("--ref_sr", type=float, default=DEFAULT_REF_SR)
    parser.add_argument("--infer_chunk_ms", type=float, default=DEFAULT_INFER_CHUNK_MS)
    parser.add_argument("--context_ms", type=float, default=DEFAULT_CONTEXT_MS)
    parser.add_argument("--audio_len", type=int, default=0, help="Fixed mixture length (0=auto)")
    parser.add_argument("--ref_frames", type=int, default=0, help="Fixed ref time dim (0=auto)")
    parser.add_argument("--image_size", type=int, default=DEFAULT_IMAGE_SIZE)
    parser.add_argument("--optimization_level", type=int, default=0)
    parser.add_argument("--min_avail_gb", type=float, default=36.0)
    args = parser.parse_args()

    require_avail_gb(args.min_avail_gb, "AV-Mossformer RKNN")

    audio_len = args.audio_len or default_audio_len(
        args.audio_sr, args.infer_chunk_ms, args.context_ms
    )
    ref_frames = args.ref_frames or default_ref_frames(audio_len, args.audio_sr, args.ref_sr)
    h = w = args.image_size

    onnx_path = os.path.abspath(args.model)
    if not os.path.isfile(onnx_path):
        print(f"ONNX not found: {onnx_path}", file=sys.stderr)
        return 1

    out_path = args.output_path
    if not out_path:
        base, _ = os.path.splitext(onnx_path)
        out_path = base + ".rknn"
    out_path = os.path.abspath(out_path)

    try:
        from rknn.api import RKNN
    except ImportError:
        print("rknn-toolkit2 is required.", file=sys.stderr)
        return 1

    rknn = RKNN(verbose=False)
    print(
        f"--> config target={args.target} mixture=[1,{audio_len}] "
        f"ref=[1,{ref_frames},{h},{w},3]"
    )
    rknn.config(target_platform=args.target, optimization_level=args.optimization_level)

    ret = rknn.load_onnx(
        model=onnx_path,
        inputs=["mixture", "ref"],
        input_size_list=[[1, audio_len], [1, ref_frames, h, w, 3]],
    )
    if ret != 0:
        print("load_onnx failed", file=sys.stderr)
        return ret

    print("--> build")
    ret = rknn.build(do_quantization=(args.dtype == "i8"))
    if ret != 0:
        print("build failed", file=sys.stderr)
        return ret

    print("--> export_rknn", out_path)
    ret = rknn.export_rknn(out_path)
    rknn.release()
    if ret != 0:
        print("export_rknn failed", file=sys.stderr)
        return ret

    print("done:", out_path)
    print(f"  fixed audio_len={audio_len} ref_frames={ref_frames} image_size={h}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
