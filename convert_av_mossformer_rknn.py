#!/usr/bin/env python3
"""Convert AV-Mossformer separator ONNX to RKNN (split deploy: ref_encoder on CPU/ORT).

INT8 calibration pipeline (see scripts/build_rknn_sep_calib.py):
  1) target_speaker_extraction_online/data/sample_2mix_calib_pairs.py --num 200
  2) scripts/build_rknn_sep_calib.py --pairs-dir checkpoints/AV_Mossformer/rknn_calib_src_200
  3) convert_av_mossformer_rknn.py --dtype i8 --dataset checkpoints/AV_Mossformer/rknn_calib_sep/dataset.txt
"""

from __future__ import annotations

import argparse
import math
import os
import sys


def _find_asr_scripts_dir() -> str:
    """Resolve asr_frontend/scripts whether this file lives under AV_TSE/ or av_tse/scripts/."""
    here = os.path.dirname(os.path.abspath(__file__))
    for rel in (
        os.path.join("asr_frontend", "scripts"),
        os.path.join("..", "..", "asr_frontend", "scripts"),
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

DEFAULT_ONNX = "./checkpoints/AV_Mossformer/av_mossformer_sep_rknn.onnx"
DEFAULT_REF_FEAT_CH = 96
DEFAULT_AUDIO_SR = 16000
DEFAULT_REF_SR = 30.0
DEFAULT_CONTEXT_MS = 100.0
DEFAULT_INFER_CHUNK_MS = 200.0


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


def _onnx_input_shapes(onnx_path: str) -> dict:
    import onnx

    m = onnx.load(onnx_path)
    shapes = {}
    for inp in m.graph.input:
        dims = []
        for d in inp.type.tensor_type.shape.dim:
            if d.dim_value > 0:
                dims.append(int(d.dim_value))
            else:
                dims.append(-1)
        shapes[inp.name] = dims
    return shapes


def main() -> int:
    parser = argparse.ArgumentParser(
        description="AV-Mossformer sep ONNX -> RKNN (ref_encoder stays on CPU/ORT)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_ONNX,
        help="Separator ONNX (export_onnx.py --export_rknn_split --sep_out ...)",
    )
    parser.add_argument("--output_path", type=str, default="", help="Output .rknn path")
    parser.add_argument("--target", type=str, default="rk3588", help="RKNN target platform")
    parser.add_argument("--dtype", type=str, default="fp", choices=("fp", "i8"))
    parser.add_argument("--audio_sr", type=int, default=DEFAULT_AUDIO_SR)
    parser.add_argument("--ref_sr", type=float, default=DEFAULT_REF_SR)
    parser.add_argument("--infer_chunk_ms", type=float, default=DEFAULT_INFER_CHUNK_MS)
    parser.add_argument("--context_ms", type=float, default=DEFAULT_CONTEXT_MS)
    parser.add_argument("--audio_len", type=int, default=4800, help="Fixed mixture length (0=auto)")
    parser.add_argument("--ref_frames", type=int, default=9, help="Fixed ref_feat time dim (0=auto)")
    parser.add_argument("--ref_feat_channels", type=int, default=DEFAULT_REF_FEAT_CH)
    parser.add_argument("--optimization_level", type=int, default=0)
    parser.add_argument("--min_avail_gb", type=float, default=36.0)
    parser.add_argument("--verbose", action="store_true", help="RKNN verbose log")
    parser.add_argument(
        "--allow_scatter",
        action="store_true",
        help="Allow ScatterElements in sep ONNX (default: require scatter-free conv export)",
    )
    parser.add_argument(
        "--ola_conv_export",
        action="store_true",
        help="Hint: re-export sep with export_onnx.py --export_rknn_split and use_ola_conv (see export_sep_rknn_onnx)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="",
        help="Calibration dataset.txt for --dtype i8 (scripts/build_rknn_sep_calib.py)",
    )
    args = parser.parse_args()

    require_avail_gb(args.min_avail_gb, "AV-Mossformer RKNN sep")

    audio_len = args.audio_len or default_audio_len(
        args.audio_sr, args.infer_chunk_ms, args.context_ms
    )
    ref_frames = args.ref_frames or default_ref_frames(audio_len, args.audio_sr, args.ref_sr)
    ref_ch = int(args.ref_feat_channels)

    onnx_path = os.path.abspath(args.model)
    if not os.path.isfile(onnx_path):
        print(f"ONNX not found: {onnx_path}", file=sys.stderr)
        print("Export first: python export_onnx.py --export_rknn_split", file=sys.stderr)
        return 1

    try:
        import onnx
        from collections import Counter

        m = onnx.load(onnx_path)
        n_nodes = len(m.graph.node)
        ops = Counter(n.op_type for n in m.graph.node)
        in_shapes = _onnx_input_shapes(onnx_path)
        print(
            "[onnx check] nodes={} inputs={} Einsum={} ScatterElements={} ConvTranspose={}".format(
                n_nodes,
                in_shapes,
                ops.get("Einsum", 0),
                ops.get("ScatterElements", 0),
                ops.get("ConvTranspose", 0),
            ),
            flush=True,
        )
        if ops.get("ScatterElements", 0) > 0 and not args.allow_scatter:
            print(
                "[onnx check] ERROR: ScatterElements in sep ONNX (decoder scatter OLA). "
                "For RKNN use export_sep_rknn_onnx(..., use_ola_conv=True) then convert again, "
                "or pass --allow_scatter to try scatter ONNX (often fails at build).",
                file=sys.stderr,
                flush=True,
            )
            return 1
        if ops.get("ScatterElements", 0) > 0:
            print(
                "[onnx check] WARN: ScatterElements present; RKNN build may fail (No lowering).",
                flush=True,
            )
        if "ref" in in_shapes and "ref_feat" not in in_shapes:
            print(
                "[onnx check] ERROR: full-model ONNX (5D ref). Use av_mossformer_sep_fixed.onnx from --export_rknn_split.",
                file=sys.stderr,
                flush=True,
            )
            return 1
    except ImportError:
        in_shapes = {"mixture": [1, audio_len], "ref_feat": [1, ref_ch, ref_frames]}

    if "ref_feat" in in_shapes:
        rf = in_shapes["ref_feat"]
        if len(rf) >= 3 and rf[1] > 0:
            ref_ch = int(rf[1])
        if len(rf) >= 3 and rf[2] > 0:
            ref_frames = int(rf[2])

    out_path = args.output_path
    if not out_path:
        base, _ = os.path.splitext(onnx_path)
        out_path = base + ("_i8.rknn" if args.dtype == "i8" else ".rknn")
    out_path = os.path.abspath(out_path)

    try:
        from rknn.api import RKNN
    except ImportError:
        print("rknn-toolkit2 is required.", file=sys.stderr)
        return 1

    rknn = RKNN(verbose=bool(args.verbose))
    print(
        f"--> config target={args.target} mixture=[1,{audio_len}] "
        f"ref_feat=[1,{ref_ch},{ref_frames}]",
        flush=True,
    )
    print(
        f"--> ref_encoder runs on CPU/ORT (gray ref); RKNN only compiles separator.",
        flush=True,
    )
    rknn.config(
        target_platform=args.target,
        optimization_level=args.optimization_level,
        single_core_mode=True,
        compress_weight=True,
    )

    print("--> load_onnx", onnx_path, flush=True)
    ret = rknn.load_onnx(
        model=onnx_path,
        inputs=["mixture", "ref_feat"],
        input_size_list=[[1, audio_len], [1, ref_ch, ref_frames]],
    )
    if ret != 0:
        print("load_onnx failed", file=sys.stderr)
        return ret
    print("--> load_onnx done", flush=True)

    do_quant = args.dtype == "i8"
    if do_quant:
        dataset = os.path.abspath(args.dataset) if args.dataset else ""
        if not dataset or not os.path.isfile(dataset):
            print(
                "ERROR: --dtype i8 requires --dataset "
                "(run: python scripts/build_rknn_sep_calib.py --wav ... --mp4 ...)",
                file=sys.stderr,
            )
            return 1
        print(f"--> build (INT8) dataset={dataset}", flush=True)
        ret = rknn.build(do_quantization=True, dataset=dataset)
    else:
        print("--> build (FP)", flush=True)
        ret = rknn.build(do_quantization=False)
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
    print(f"  audio_len={audio_len} ref_feat=[1,{ref_ch},{ref_frames}]")
    print(f"  pair with ORT: checkpoints/AV_Mossformer/av_mossformer_ref_fixed.onnx")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
