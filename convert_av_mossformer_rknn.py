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

DEFAULT_ONNX = "./checkpoints/AV_Mossformer/av_mossformer2_fixed.onnx"
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
    parser.add_argument("--audio_len", type=int, default=9600, help="Fixed mixture length (0=auto)")
    parser.add_argument("--ref_frames", type=int, default=18, help="Fixed ref time dim (0=auto)")
    parser.add_argument("--image_size", type=int, default=DEFAULT_IMAGE_SIZE)
    parser.add_argument("--optimization_level", type=int, default=0)
    parser.add_argument(
        "--no_input_size",
        action="store_true",
        help="load_onnx without input_size_list (often reaches OpEmit; try if build aborts)",
    )
    parser.add_argument(
        "--sep_only",
        action="store_true",
        help="ONNX has output sep_pack (export with --export_part sep)",
    )
    parser.add_argument(
        "--ref_is_gray",
        action="store_true",
        help="ONNX ref input is 4D grayscale [B,T,H,W] (export with --ref_is_gray)",
    )
    parser.add_argument("--min_avail_gb", type=float, default=36.0)
    parser.add_argument(
        "--single_core_mode",
        action="store_true",
        default=True,
        help="Build for single NPU core (reduces LayoutMatch work by 2/3)",
    )
    parser.add_argument(
        "--compress_weight",
        action="store_true",
        default=True,
        help="Compress weights during build (reduces memory pressure)",
    )
    parser.add_argument(
        "--remove_weight",
        action="store_true",
        default=True,
        help="Remove weights from RKNN graph layout computation",
    )
    parser.add_argument(
        "--remove_reshape",
        action="store_true",
        default=True,
        help="DISABLE Reshape op optimization (LayoutMatch crash fix)",
    )
    parser.add_argument(
        "--disable_reshape_rules",
        action="store_true",
        default=True,
        help="Disable Reshape fusion rules to prevent ref tensor auto-fusion crash",
    )
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

    try:
        import onnx
        from collections import Counter

        m = onnx.load(onnx_path)
        n_nodes = len(m.graph.node)
        ops = Counter(n.op_type for n in m.graph.node)
        print(
            "[onnx check] nodes={} Einsum={} ScatterElements={} EyeLike={} ConvTranspose={} Resize={}".format(
                n_nodes,
                ops.get("Einsum", 0),
                ops.get("ScatterElements", 0),
                ops.get("EyeLike", 0),
                ops.get("ConvTranspose", 0),
                ops.get("Resize", 0),
            ),
            flush=True,
        )
        size_mb = os.path.getsize(onnx_path) / (1024 * 1024)
        if size_mb > 100:
            print(
                f"[onnx check] WARN: ONNX file is {size_mb:.1f} MB (likely dense OLA matrix). "
                "Re-export with scatter OLA (current av_mossformer2.py).",
                flush=True,
            )
        elif n_nodes > 50000:
            print(
                f"[onnx check] WARN: graph has {n_nodes} nodes (decoder loop unroll?). "
                "Re-export with export_onnx.py --fixed.",
                flush=True,
            )
        for n in m.graph.node:
            if n.op_type == "ConvTranspose":
                attrs = [a.name for a in n.attribute]
                print(f"[onnx check] ConvTranspose attrs: {attrs}", flush=True)
                break
    except ImportError:
        pass

    out_path = args.output_path
    if not out_path:
        base, _ = os.path.splitext(onnx_path)
        out_path = base + ".rknn"
    out_path = os.path.abspath(out_path)

    # Log file path
    log_file = out_path.replace('.rknn', '_build.log')
    print(f"--> logging to {log_file}", flush=True)

    try:
        from rknn.api import RKNN
    except ImportError:
        print("rknn-toolkit2 is required.", file=sys.stderr)
        return 1

    rknn = RKNN(verbose=True, verbose_file=log_file)
    print(
        f"--> config target={args.target} mixture=[1,{audio_len}] "
        f"ref=[1,{ref_frames},{h},{w},3]",
        flush=True,
    )
    print(
        f"--> ref_frames={ref_frames} must match ONNX export T_ref "
        f"(export_onnx compute_stream_window_lengths)",
        flush=True,
    )
    disable_rules = None
    if args.disable_reshape_rules:
        disable_rules = [
            "Remove_Useless_Reshape",
            "Remove_Redundant_Slice",
            "Fuse_Reshape_Into_Previous",
            "Fuse_Reshape_Into_Next",
            "Fuse_Slice_Into_Reshape",
        ]
    # Use optimization_level=-1 to disable ALL optimizations
    rknn.config(
        target_platform=args.target,
        optimization_level=-1,
        single_core_mode=args.single_core_mode,
    )

    print("--> load_onnx", onnx_path, flush=True)
    load_kw = dict(model=onnx_path)
    if not args.no_input_size:
        load_kw["inputs"] = ["mixture", "ref"]
        load_kw["input_size_list"] = [[1, audio_len], [1, ref_frames, h, w, 3] if not args.ref_is_gray else [1, ref_frames, h, w]]
    if args.sep_only:
        kernel_size = 16
        stride = max(1, kernel_size // 2)
        t_enc = (int(audio_len) - kernel_size) // stride + 1
        n_ch = 512
        try:
            import onnx

            m = onnx.load(onnx_path)
            for vi in list(m.graph.input) + list(m.graph.value_info):
                if "mixture_w" in vi.name and vi.type.tensor_type.shape.dim:
                    for d in vi.type.tensor_type.shape.dim[1:2]:
                        if d.dim_value:
                            n_ch = int(d.dim_value)
        except Exception:
            pass
        print(
            f"[sep_only] ONNX output sep_pack; expect channels~{2 * n_ch}, T_enc={t_enc}",
            flush=True,
        )
    ret = rknn.load_onnx(**load_kw)
    if ret != 0:
        print("load_onnx failed", file=sys.stderr)
        return ret
    print("--> load_onnx done", flush=True)

    print("--> build", flush=True)
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
