#!/usr/bin/env python3
"""Hybrid inference: RKNN separator + ONNX Runtime decoder (split deploy)."""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def main() -> int:
    parser = argparse.ArgumentParser(description="RKNN sep + ORT decoder hybrid run")
    parser.add_argument("--rknn", required=True, help="Sep-only .rknn from convert --sep_only")
    parser.add_argument("--decoder_onnx", required=True, help="Decoder ONNX (--export_part decoder)")
    parser.add_argument("--mixture_npy", required=True, help="[1, T_audio] float32")
    parser.add_argument("--ref_npy", required=True, help="[1, T_ref, H, W, 3] float32")
    parser.add_argument("--target", default="rk3588")
    parser.add_argument("--out_npy", default="hybrid_out.npy")
    args = parser.parse_args()

    mixture = np.load(args.mixture_npy).astype(np.float32)
    ref = np.load(args.ref_npy).astype(np.float32)

    from rknn.api import RKNN

    rknn = RKNN(verbose=False)
    if rknn.load_rknn(args.rknn) != 0:
        print("load_rknn failed", file=sys.stderr)
        return 1
    if rknn.init_runtime(target=args.target) != 0:
        print("init_runtime failed", file=sys.stderr)
        return 1
    pack = rknn.inference(inputs=[mixture, ref])[0]
    rknn.release()
    pack = np.asarray(pack)
    c = pack.shape[1] // 2
    mw, mask = pack[:, :c, :], pack[:, c:, :]

    import onnxruntime as ort

    sess = ort.InferenceSession(args.decoder_onnx, providers=["CPUExecutionProvider"])
    tlen = np.array([int(mixture.shape[-1])], dtype=np.int64)
    out = sess.run(
        None,
        {
            "mixture_w": mw.astype(np.float32),
            "est_mask": mask.astype(np.float32),
            "target_audio_len": tlen,
        },
    )[0]
    np.save(args.out_npy, out)
    print("saved", args.out_npy, "shape", out.shape)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
