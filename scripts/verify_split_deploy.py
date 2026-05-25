#!/usr/bin/env python3
"""Verify split deploy: ORT ref_encoder + ORT sep vs full ONNX; optional RKNN sep file check."""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from export_onnx import compute_stream_window_lengths, verify_rknn_split_onnx  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ref_onnx",
        default=os.path.join(ROOT, "checkpoints/AV_Mossformer/av_mossformer_ref_fixed.onnx"),
    )
    parser.add_argument(
        "--sep_onnx",
        default=os.path.join(ROOT, "checkpoints/AV_Mossformer/av_mossformer_sep_fixed.onnx"),
    )
    parser.add_argument(
        "--full_onnx",
        default=os.path.join(ROOT, "checkpoints/AV_Mossformer/av_mossformer2_fixed.onnx"),
    )
    parser.add_argument(
        "--sep_rknn",
        default=os.path.join(ROOT, "checkpoints/AV_Mossformer/av_mossformer_sep_rknn.rknn"),
    )
    parser.add_argument("--checkpoint", default=os.path.join(ROOT, "checkpoints/AV_Mossformer/last_best_weights_only.pt"))
    args = parser.parse_args()

    t_audio, t_ref = compute_stream_window_lengths()
    from export_onnx import build_model, load_config

    cfg = load_config(os.path.join(ROOT, "checkpoints/AV_Mossformer/config.yaml"))
    model = build_model(cfg, args.checkpoint)
    verify_rknn_split_onnx(
        args.ref_onnx,
        args.sep_onnx,
        model=model,
        t_audio=t_audio,
        t_ref=t_ref,
    )

    sep_rknn = os.path.join(ROOT, "checkpoints/AV_Mossformer/av_mossformer_sep_rknn.onnx")
    if os.path.isfile(sep_rknn):
        import onnx

        ops = __import__("collections").Counter(
            n.op_type for n in onnx.load(sep_rknn).graph.node
        )
        print(f"[check] sep_rknn ScatterElements={ops.get('ScatterElements', 0)}")

    if os.path.isfile(args.full_onnx):
        import onnxruntime as ort
        from networks import network_wrapper

        mix = np.random.randn(1, t_audio).astype(np.float32)
        ref_rgb = np.random.randn(1, t_ref, 96, 96, 3).astype(np.float32)
        full_sess = ort.InferenceSession(args.full_onnx, providers=["CPUExecutionProvider"])
        full_out = full_sess.run(None, {"mixture": mix, "ref": ref_rgb})[0].squeeze()

        from av_stream_inference import _SplitOnnxModelWrapper

        split = _SplitOnnxModelWrapper(args.ref_onnx, args.sep_onnx, image_size=96)
        split_out = split.call_numpy(mix, ref_rgb).squeeze()
        diff = np.abs(full_out - split_out).max()
        print(f"[split vs full ONNX] max|diff| = {diff:.6e}")
        if diff < 1e-4:
            print("[ok] split ORT matches legacy full ONNX")
        else:
            print(f"[warn] full ONNX may be stale vs split (diff={diff:.4f})")

    if os.path.isfile(args.sep_rknn):
        mb = os.path.getsize(args.sep_rknn) / (1024 * 1024)
        print(f"[ok] RKNN sep exists: {args.sep_rknn} ({mb:.1f} MB)")
    else:
        print(f"[warn] RKNN sep not found: {args.sep_rknn}")

    print("Split deploy verification passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
