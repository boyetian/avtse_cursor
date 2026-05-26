#!/usr/bin/env python3
"""List ONNX Mul/Div/Reshape nodes whose tensor element count may exceed RKNN's ~8191 limit."""

from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

RKNN_FLATTEN_LIMIT = 8191


def _shape_from_value_info(vi) -> list[int] | None:
    tt = vi.type.tensor_type
    if not tt.HasField("shape"):
        return None
    dims: list[int] = []
    for d in tt.shape.dim:
        if d.dim_value > 0:
            dims.append(int(d.dim_value))
        elif d.dim_param:
            return None
        else:
            return None
    return dims


def _elem_count(shape: list[int]) -> int:
    n = 1
    for s in shape:
        n *= int(s)
    return n


def _broadcast_product(shapes: list[list[int]]) -> int | None:
    """Approximate element count after numpy-style broadcast (unknown dim -> skip)."""
    if not shapes:
        return None
    ndim = max(len(s) for s in shapes)
    out: list[int] = []
    for i in range(ndim):
        sizes = []
        for s in shapes:
            if len(s) < ndim:
                s = [1] * (ndim - len(s)) + s
            sizes.append(s[-(ndim - i)])
        if any(x < 0 for x in sizes):
            return None
        if 1 in sizes:
            out.append(max(sizes))
        elif len(set(sizes)) == 1:
            out.append(sizes[0])
        else:
            return None
    return _elem_count(out)


def diagnose(onnx_path: str, limit: int = RKNN_FLATTEN_LIMIT) -> int:
    import onnx
    from onnx import numpy_helper

    model = onnx.load(onnx_path)
    graph = model.graph

    name_to_shape: dict[str, list[int]] = {}
    for vi in list(graph.input) + list(graph.value_info) + list(graph.output):
        sh = _shape_from_value_info(vi)
        if sh is not None:
            name_to_shape[vi.name] = sh

    for init in graph.initializer:
        t = numpy_helper.to_array(init)
        name_to_shape[init.name] = list(t.shape)

    op_types = ("Mul", "Div", "Reshape")
    hits: list[tuple[str, str, int, str]] = []

    for node in graph.node:
        if node.op_type not in op_types:
            continue
        in_shapes: list[list[int]] = []
        for inp in node.input:
            if inp and inp in name_to_shape:
                in_shapes.append(name_to_shape[inp])
        prod = _broadcast_product(in_shapes) if node.op_type in ("Mul", "Div") else None
        if prod is None and in_shapes:
            prod = max(_elem_count(s) for s in in_shapes)
        if prod is not None and prod > limit:
            shape_str = ", ".join(str(s) for s in in_shapes)
            hits.append((node.op_type, node.name or "(unnamed)", prod, shape_str))

        for out in node.output:
            if out in name_to_shape:
                c = _elem_count(name_to_shape[out])
                if c > limit:
                    hits.append(
                        (
                            node.op_type,
                            node.name or "(unnamed)",
                            c,
                            f"output {out} shape={name_to_shape[out]}",
                        )
                    )

    ops = defaultdict(int)
    for n in graph.node:
        ops[n.op_type] += 1

    print(f"ONNX: {onnx_path}")
    print(f"  nodes={len(graph.node)} Mul={ops['Mul']} Div={ops['Div']} Reshape={ops['Reshape']}")
    print(f"  tensors with known shape in graph: {len(name_to_shape)}")
    print(f"  suspect ops (element count > {limit}):")

    if not hits:
        print("    (none found with static shapes)")
        return 0

    seen = set()
    for op, name, prod, detail in sorted(hits, key=lambda x: -x[2]):
        key = (op, name, prod)
        if key in seen:
            continue
        seen.add(key)
        print(f"    {op} {name}: ~{prod} elements  ({detail})")
    return len(seen)


def main() -> int:
    parser = argparse.ArgumentParser(description="Diagnose sep ONNX for RKNN flatten limits")
    parser.add_argument(
        "onnx_path",
        nargs="?",
        default=os.path.join(ROOT, "checkpoints", "AV_Mossformer", "av_mossformer_sep_rknn.onnx"),
    )
    parser.add_argument("--limit", type=int, default=RKNN_FLATTEN_LIMIT)
    args = parser.parse_args()
    path = os.path.abspath(args.onnx_path)
    if not os.path.isfile(path):
        print(f"ONNX not found: {path}", file=sys.stderr)
        return 1
    diagnose(path, limit=args.limit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
