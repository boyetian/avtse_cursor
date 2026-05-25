#!/usr/bin/env python3
"""Convert DFSMN AEC ONNX to RKNN for RK3588 (and other RKNPU2 platforms)."""

from __future__ import annotations

import argparse
import os
import sys
import tempfile

from onnx_patch_dfsmn_for_rknn import patch_dfsmn_for_rknn
from rknn_mem_guard import require_avail_gb

# Fixed chunk size aligned with cpp/asr_frontend dfsmn_aec_onnx.hpp
CHUNK_SIZE = 16320


def main() -> int:
    parser = argparse.ArgumentParser(description="DFSMN AEC ONNX -> RKNN")
    parser.add_argument(
        "--model",
        type=str,
        default="../../../asr_frontend/checkpoints/dfsmn_aec_16k/DFSMN_AEC_opt.onnx",
        help="Path to DFSMN_AEC_opt.onnx",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="",
        help="Output .rknn path (default: same dir as ONNX with .rknn suffix)",
    )
    parser.add_argument("--target", type=str, default="rk3588", help="RKNN target platform")
    parser.add_argument(
        "--dtype",
        type=str,
        default="fp",
        choices=("fp", "i8"),
        help="fp: no quantization; i8: quantized (needs calibration dataset)",
    )
    parser.add_argument(
        "--min_avail_gb",
        type=float,
        default=8.0,
        help="Exit without converting if MemAvailable is below this (GiB)",
    )
    args = parser.parse_args()

    require_avail_gb(args.min_avail_gb, "DFSMN AEC RKNN")

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
        print(
            "rknn-toolkit2 is required. Install per Rockchip docs, then re-run.",
            file=sys.stderr,
        )
        return 1

    with tempfile.NamedTemporaryFile(suffix="_float_io.onnx", delete=False) as tmp:
        patched_onnx = tmp.name
    print("--> patch ONNX (int16 IO -> float for RKNN toolkit)")
    patch_dfsmn_for_rknn(onnx_path, patched_onnx)

    rknn = RKNN(verbose=True)
    print("--> config")
    rknn.config(target_platform=args.target)

    print("--> load_onnx (near_end_audio / far_end_audio -> [1, 1, {}])".format(CHUNK_SIZE))
    ret = rknn.load_onnx(
        model=patched_onnx,
        inputs=["near_end_audio", "far_end_audio"],
        input_size_list=[[1, 1, CHUNK_SIZE], [1, 1, CHUNK_SIZE]],
    )
    if ret != 0:
        print("load_onnx failed", file=sys.stderr)
        return ret

    print("--> build")
    do_quant = args.dtype == "i8"
    ret = rknn.build(do_quantization=do_quant)
    if ret != 0:
        print("build failed (check unsupported ops / ai.onnx.ml)", file=sys.stderr)
        return ret

    print("--> export_rknn", out_path)
    ret = rknn.export_rknn(out_path)
    rknn.release()
    if ret != 0:
        print("export_rknn failed", file=sys.stderr)
        return ret

    try:
        os.remove(patched_onnx)
    except OSError:
        pass
    print("done:", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
