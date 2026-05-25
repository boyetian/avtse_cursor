#!/usr/bin/env python3
"""Patch DFSMN AEC ONNX: int16 IO -> float for rknn-toolkit2 (bypass input/output Cast)."""

from __future__ import annotations

import argparse
import os
import sys

import onnx
from onnx import TensorProto, helper


def _node_copy_with_inputs(node: onnx.NodeProto, inputs: list[str]) -> onnx.NodeProto:
    attrs = {a.name: helper.get_attribute_value(a) for a in node.attribute}
    return helper.make_node(
        node.op_type, inputs, list(node.output), name=node.name, **attrs
    )


def patch_dfsmn_for_rknn(src_path: str, dst_path: str) -> None:
    model = onnx.load(src_path)
    graph = model.graph

    # Input: int16 -> float; bypass Cast + Mul(scale=1/32768) when feeding normalized float.
    input_bypass: dict[str, str] = {}
    remove_names: set[str] = set()
    for node in graph.node:
        if node.op_type == "Cast" and len(node.input) == 1 and node.input[0] in (
            "near_end_audio",
            "far_end_audio",
        ):
            cast_out = node.output[0]
            for n2 in graph.node:
                if n2.op_type == "Mul" and cast_out in n2.input:
                    mul_out = n2.output[0]
                    input_bypass[mul_out] = node.input[0]
                    remove_names.add(node.name)
                    remove_names.add(n2.name)
                    break

    # Output: drop Cast to int16; rename producer tensor to `aec_audio`.
    output_clip_tensor: str | None = None
    for node in graph.node:
        if (
            node.op_type == "Cast"
            and node.output
            and node.output[0] == "aec_audio"
        ):
            if node.input:
                output_clip_tensor = node.input[0]
                remove_names.add(node.name)

    new_nodes: list[onnx.NodeProto] = []
    for node in graph.node:
        if node.name in remove_names:
            continue
        new_in = []
        for x in node.input:
            if x in input_bypass:
                new_in.append(input_bypass[x])
            else:
                new_in.append(x)
        new_out = list(node.output)
        if output_clip_tensor and output_clip_tensor in new_out:
            new_out = ["aec_audio" if t == output_clip_tensor else t for t in new_out]
        if new_in != list(node.input) or new_out != list(node.output):
            attrs = {a.name: helper.get_attribute_value(a) for a in node.attribute}
            new_nodes.append(
                helper.make_node(
                    node.op_type, new_in, new_out, name=node.name, **attrs
                )
            )
        else:
            new_nodes.append(node)

    if output_clip_tensor and output_clip_tensor != "aec_audio":
        for node in new_nodes:
            new_in = [
                "aec_audio" if x == output_clip_tensor else x for x in node.input
            ]
            if new_in != list(node.input):
                attrs = {a.name: helper.get_attribute_value(a) for a in node.attribute}
                idx = new_nodes.index(node)
                new_nodes[idx] = helper.make_node(
                    node.op_type,
                    new_in,
                    list(node.output),
                    name=node.name,
                    **attrs,
                )

    del graph.node[:]
    graph.node.extend(new_nodes)

    # rknn-toolkit2 re-exports LSTM with `layout`; bundled onnx checker rejects it.
    for node in graph.node:
        if node.op_type != "LSTM":
            continue
        kept = [a for a in node.attribute if a.name != "layout"]
        if len(kept) != len(node.attribute):
            del node.attribute[:]
            node.attribute.extend(kept)

    for inp in graph.input:
        if inp.name in ("near_end_audio", "far_end_audio"):
            inp.type.tensor_type.elem_type = TensorProto.FLOAT

    for out in graph.output:
        if out.name == "aec_audio":
            out.type.tensor_type.elem_type = TensorProto.FLOAT

    os.makedirs(os.path.dirname(os.path.abspath(dst_path)) or ".", exist_ok=True)
    onnx.checker.check_model(model)
    onnx.save(model, dst_path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    patch_dfsmn_for_rknn(os.path.abspath(args.model), os.path.abspath(args.output))
    print("patched:", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
