#!/usr/bin/env python3
"""Strip a full training checkpoint to model weights only (no optimizer, scheduler, etc.)."""

from __future__ import annotations

import argparse
import inspect
import sys
from typing import Any, Dict, Mapping

import torch


def _is_state_dict_like(d: Mapping[str, Any]) -> bool:
    if not d:
        return False
    for v in d.values():
        if not torch.is_tensor(v):
            return False
    return True


def _to_cpu_state_dict(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {k: v.detach().cpu() for k, v in sd.items()}


def extract_model_state_dict(ckpt: Any) -> Dict[str, torch.Tensor]:
    if isinstance(ckpt, dict):
        if "model" in ckpt:
            m = ckpt["model"]
            if isinstance(m, torch.nn.Module):
                return _to_cpu_state_dict(m.state_dict())
            if isinstance(m, dict) and _is_state_dict_like(m):
                return _to_cpu_state_dict(dict(m))
        if "state_dict" in ckpt:
            sd = ckpt["state_dict"]
            if isinstance(sd, dict) and _is_state_dict_like(sd):
                return _to_cpu_state_dict(dict(sd))
        if _is_state_dict_like(ckpt):
            return _to_cpu_state_dict(dict(ckpt))

    raise ValueError(
        "Unrecognized checkpoint structure. Top-level type: %s. "
        "If dict, keys: %s"
        % (
            type(ckpt).__name__,
            list(ckpt.keys()) if isinstance(ckpt, dict) else None,
        )
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--in",
        dest="inp",
        default="checkpoints/AV_Mossformer/last_best_checkpoint.pt",
        help="Input full checkpoint path",
    )
    p.add_argument(
        "--out",
        dest="out",
        default="checkpoints/AV_Mossformer/last_best_weights_only.pt",
        help="Output weights-only path",
    )
    args = p.parse_args()

    load_kw: Dict[str, Any] = {"map_location": "cpu"}
    if "weights_only" in inspect.signature(torch.load).parameters:
        load_kw["weights_only"] = False
    ckpt = torch.load(args.inp, **load_kw)
    try:
        model_sd = extract_model_state_dict(ckpt)
    except ValueError as e:
        print(e, file=sys.stderr)
        sys.exit(1)

    torch.save({"model": model_sd}, args.out)
    print("Saved %d tensors to %s" % (len(model_sd), args.out))


if __name__ == "__main__":
    main()
