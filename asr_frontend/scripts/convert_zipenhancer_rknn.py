#!/usr/bin/env python3
"""Convert ZipEnhancer full ONNX to RKNN with fixed streaming window shape."""

from __future__ import annotations

import argparse
import os
import sys

from rknn_mem_guard import require_avail_gb

DEFAULT_SAMPLE_RATE = 16000
DEFAULT_DECODE_WINDOW_SEC = 1.0


def main() -> int:
    parser = argparse.ArgumentParser(description="ZipEnhancer full ONNX -> RKNN")
    parser.add_argument(
        "--model",
        type=str,
        default="../../../asr_frontend/asr_frontend/checkpoints/zip_enhancer_se_16k/zipenhancer_full_sim.onnx",
        help="Path to zipenhancer_full_sim.onnx (run onnxsim on zipenhancer_full.onnx first)",
    )
    parser.add_argument("--output_path", type=str, default="", help="Output .rknn path")
    parser.add_argument("--target", type=str, default="rk3588", help="RKNN target platform")
    parser.add_argument(
        "--dtype",
        type=str,
        default="fp",
        choices=("fp", "i8"),
        help="fp: no quantization; i8: quantized",
    )
    parser.add_argument(
        "--decode_window_sec",
        type=float,
        default=DEFAULT_DECODE_WINDOW_SEC,
        help="Streaming window in seconds (must match C++ se_decode_window_sec)",
    )
    parser.add_argument(
        "--sample_rate",
        type=int,
        default=DEFAULT_SAMPLE_RATE,
        help="Sample rate in Hz",
    )
    parser.add_argument(
        "--optimization_level",
        type=int,
        default=0,
        help="RKNN graph optimization level (0 uses less RAM during build)",
    )
    parser.add_argument(
        "--single_core_mode",
        action="store_true",
        default=True,
        help="Build for single NPU core only (reduces peak build RAM ~2/3 on rk3588's 3-core NPU)",
    )
    parser.add_argument(
        "--no_single_core_mode",
        dest="single_core_mode",
        action="store_false",
        help="Disable single_core_mode (build for all NPU cores)",
    )
    parser.add_argument(
        "--compress_weight",
        action="store_true",
        default=True,
        help="Compress weight tensors in the RKNN model (reduces build and runtime memory)",
    )
    parser.add_argument(
        "--no_compress_weight",
        dest="compress_weight",
        action="store_false",
        help="Disable weight compression",
    )
    parser.add_argument(
        "--min_avail_gb",
        type=float,
        default=16.0,
        help="Exit without converting if MemAvailable is below this (GiB)",
    )
    args = parser.parse_args()

    require_avail_gb(args.min_avail_gb, "ZipEnhancer RKNN")

    window = int(args.sample_rate * args.decode_window_sec)
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

    rknn = RKNN(verbose=False)
    print(
        f"--> config (fixed input shape [1, {window}], opt_level={args.optimization_level}, "
        f"single_core={args.single_core_mode}, compress_weight={args.compress_weight})"
    )
    rknn.config(
        target_platform=args.target,
        optimization_level=args.optimization_level,
        single_core_mode=args.single_core_mode,
        compress_weight=args.compress_weight,
    )

    print("--> load_onnx (input noisy -> [1, {}])".format(window))
    ret = rknn.load_onnx(
        model=onnx_path,
        inputs=["noisy"],
        input_size_list=[[1, window]],
    )
    if ret != 0:
        print("load_onnx failed", file=sys.stderr)
        return ret

    print("--> build")
    do_quant = args.dtype == "i8"
    ret = rknn.build(do_quantization=do_quant)
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
