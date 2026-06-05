"""Check if ONNX separator weights match torch checkpoint weights."""

import os
import sys
import numpy as np
import onnxruntime as ort
import torch
import yaml

sys.path.insert(0, os.path.dirname(__file__))
from av_stream_inference import _load_model_weights, _dict_to_ns
from networks import network_wrapper


def main():
    config_path = "./checkpoints/AV_Mossformer/config.yaml"
    checkpoint_dir = "./checkpoints/AV_Mossformer"
    sep_onnx_path = "./checkpoints/AV_Mossformer/av_mossformer_sep_fixed.onnx"

    # Load torch model
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    ns = _dict_to_ns(cfg)
    ns.device = torch.device("cpu")
    model = network_wrapper(ns)
    model.eval()
    ckpt = os.path.join(checkpoint_dir, "last_best_weights_only.pt")
    if not os.path.isfile(ckpt):
        ckpt = os.path.join(checkpoint_dir, "last_best_checkpoint.pt")
    _load_model_weights(model, ckpt)

    mossformer = model.model  # av_mossformer2
    sep = mossformer.sep_network  # Mossformer

    # Load ONNX model
    sess = ort.InferenceSession(sep_onnx_path, providers=["CPUExecutionProvider"])

    # Print ONNX input/output info
    print("=== ONNX Inputs ===")
    for inp in sess.get_inputs():
        print(f"  {inp.name}: {inp.shape}")
    print("=== ONNX Outputs ===")
    for out in sess.get_outputs():
        print(f"  {out.name}: {out.shape}")

    # Get all ONNX initializers via onnx library
    print("\n=== Weight Comparison ===")
    import onnx
    onnx_model = onnx.load(sep_onnx_path)
    onnx_init_map = {}
    for init in onnx_model.graph.initializer:
        name = init.name
        data = onnx.numpy_helper.to_array(init)
        onnx_init_map[name] = data

    print(f"ONNX initializers: {len(onnx_init_map)}")
    for name in sorted(onnx_init_map.keys())[:20]:
        print(f"  {name}: {onnx_init_map[name].shape}")

    # Compare encoder conv1d weight
    torch_enc_weight = sep.encoder.conv1d_U.weight.detach().cpu().numpy()
    print(f"\nTorch encoder.conv1d_U.weight: shape={torch_enc_weight.shape}, range=[{torch_enc_weight.min():.6f}, {torch_enc_weight.max():.6f}]")

    # Find matching ONNX weight - the name might differ due to tracing
    # Look for a weight with shape [64, 1, 16]
    found_enc = False
    for name, data in onnx_init_map.items():
        if data.shape == torch_enc_weight.shape:
            diff = np.abs(data - torch_enc_weight).max()
            print(f"  {name}: shape={data.shape}, max_diff={diff:.6e}")
            if diff < 1e-5:
                print(f"    *** MATCH ***")
                found_enc = True

    if not found_enc:
        print("  No matching encoder weight found in ONNX!")
        # List all weight shapes to help debug
        print("\n  All ONNX weight shapes:")
        shapes = {}
        for name, data in onnx_init_map.items():
            s = str(data.shape)
            shapes.setdefault(s, []).append(name)
        for s, names in sorted(shapes.items()):
            print(f"  {s}: {names[:3]}")

    # Compare decoder basis_signals weight
    torch_dec_weight = sep.decoder.basis_signals.weight.detach().cpu().numpy()
    print(f"\nTorch decoder.basis_signals.weight: shape={torch_dec_weight.shape}")

    found_dec = False
    for name, data in onnx_init_map.items():
        if data.shape == torch_dec_weight.shape:
            diff = np.abs(data - torch_dec_weight).max()
            print(f"  {name}: shape={data.shape}, max_diff={diff:.6e}")
            if diff < 1e-5:
                print(f"    *** MATCH ***")
                found_dec = True

    if not found_dec:
        print("  No matching decoder weight found in ONNX!")

    # Compare separator bottleneck weight
    torch_bn_weight = sep.separator.bottleneck_conv1x1.weight.detach().cpu().numpy()
    print(f"\nTorch separator.bottleneck_conv1x1.weight: shape={torch_bn_weight.shape}")

    found_bn = False
    for name, data in onnx_init_map.items():
        if data.shape == torch_bn_weight.shape:
            diff = np.abs(data - torch_bn_weight).max()
            print(f"  {name}: shape={data.shape}, max_diff={diff:.6e}")
            if diff < 1e-5:
                print(f"    *** MATCH ***")
                found_bn = True

    if not found_bn:
        print("  No matching bottleneck weight found in ONNX!")

    # Check if ALL torch weights have ONNX matches
    print("\n=== Full Weight Matching ===")
    torch_params = {}
    for name, param in sep.named_parameters():
        torch_params[name] = param.detach().cpu().numpy()

    matched = 0
    matched_with_diff = 0
    max_any_diff = 0.0
    unmatched = []
    for t_name, t_data in torch_params.items():
        t_shape = t_data.shape
        found = False
        best_diff = float("inf")
        best_oname = ""
        for o_name, o_data in onnx_init_map.items():
            if o_data.shape == t_shape:
                diff = np.abs(o_data - t_data).max()
                if diff < best_diff:
                    best_diff = diff
                    best_oname = o_name
                if diff < 1e-4:
                    found = True
                    matched += 1
                    if diff > 1e-7:
                        matched_with_diff += 1
                        max_any_diff = max(max_any_diff, diff)
                    break
        if not found:
            # Check transposed shapes (for Linear layers)
            if t_data.ndim == 2:
                t_shape_t = (t_shape[1], t_shape[0])
                for o_name, o_data in onnx_init_map.items():
                    if o_data.shape == t_shape_t:
                        diff = np.abs(o_data - t_data.T).max()
                        if diff < 1e-4:
                            found = True
                            matched += 1
                            best_oname = o_name + " (transposed)"
                            break
                        if diff < best_diff:
                            best_diff = diff
                            best_oname = o_name + " (transposed)"
            if not found:
                unmatched.append((t_name, t_shape, best_oname, best_diff))

    print(f"Matched: {matched}/{len(torch_params)} weights")
    if matched_with_diff > 0:
        print(f"  ({matched_with_diff} matched with diff > 1e-7, max={max_any_diff:.2e})")
    if unmatched:
        print(f"Unmatched ({len(unmatched)}):")
        for name, shape, best_oname, best_diff in unmatched[:20]:
            print(f"  {name}: shape={shape}, best_match={best_oname}, best_diff={best_diff:.2e}")


if __name__ == "__main__":
    main()
