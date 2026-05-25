"""导出 TorchScript 模型（流式 CPU / LibTorch C++ 部署）。

默认 trace 窗长与 export_onnx 定长窗一致：context + hop + lookahead
（main 默认 context_ms=100、infer_chunk_ms=500 → mixture 9600、ref 18 帧）。
"""

import argparse
import os
from typing import Optional, Set, Tuple

import numpy as np
import torch
import torch.nn as nn

from export_onnx import build_model, compute_stream_window_lengths, load_config


def _stream_shape(
    audio_sr: int,
    ref_sr: float,
    context_ms: float,
    infer_chunk_ms: float,
    lookahead_ms: float = 0.0,
) -> Tuple[int, int]:
    """稳态窗 T_audio = context + hop + lookahead，与 export_onnx / AVStreamInference 滑窗一致。"""
    return compute_stream_window_lengths(
        audio_sr=int(audio_sr),
        ref_sr=float(ref_sr),
        context_ms=float(context_ms),
        infer_chunk_ms=float(infer_chunk_ms),
        lookahead_ms=float(lookahead_ms),
    )


def _make_dummy(
    audio_sr: int,
    ref_sr: float,
    context_ms: float,
    infer_chunk_ms: float,
    lookahead_ms: float,
    image_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    t_audio, t_ref = _stream_shape(
        audio_sr, ref_sr, context_ms, infer_chunk_ms, lookahead_ms
    )
    return (
        torch.randn(1, t_audio, dtype=torch.float32),
        torch.randn(1, t_ref, image_size, image_size, 3, dtype=torch.float32),
    )


def _warmup(model, dummy_inputs, iters: int):
    for _ in range(max(0, int(iters))):
        with torch.no_grad():
            _ = model(*dummy_inputs)


def _apply_inference_optims(traced, label: str):
    try:
        traced = torch.jit.optimize_for_inference(traced)
        print(f"[{label}] applied torch.jit.optimize_for_inference")
    except Exception as e:
        print(f"[{label}] optimize_for_inference 跳过: {type(e).__name__}: {e}")
    return traced


def trace_and_save(
    model,
    dummy_inputs,
    output_path,
    label,
    also_save_path: Optional[str] = None,
):
    print(f"[{label}] Tracing ...")
    with torch.no_grad():
        traced = torch.jit.trace(model, dummy_inputs, check_trace=False)
    traced = torch.jit.freeze(traced)
    # NOTE: do NOT save optimize_for_inference graph; some PyTorch versions
    # cannot deserialize it (e.g. "required keyword attribute 'value'").
    # Apply optimize_for_inference at load-time instead.
    traced.save(output_path)
    mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"[{label}] 保存到 {output_path} ({mb:.1f} MB)")
    if also_save_path and os.path.abspath(also_save_path) != os.path.abspath(output_path):
        traced.save(also_save_path)
        mb2 = os.path.getsize(also_save_path) / (1024 * 1024)
        print(f"[{label}] 定长副本 → {also_save_path} ({mb2:.1f} MB)")
    # Return the optimized version for in-process numerical comparison only.
    return _apply_inference_optims(traced, label)


def script_and_save(model, output_path, label):
    print(f"[{label}] Scripting ...")
    scripted = torch.jit.script(model)
    scripted = torch.jit.freeze(scripted)
    scripted.save(output_path)
    mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"[{label}] 保存到 {output_path} ({mb:.1f} MB)")
    return _apply_inference_optims(scripted, label)


def _parse_skip_names(raw: str) -> Set[str]:
    parts = [x.strip() for x in str(raw).split(",")]
    return {x for x in parts if x}


def _upcast_jit_fp16_weights(model):
    """将 JIT 模型中的 FP16 参数/缓冲区上转为 FP32。"""
    for name, param in model.named_parameters():
        if param.dtype == torch.float16:
            param.data = param.data.float()
    for name, buf in model.named_buffers():
        if buf.dtype == torch.float16:
            model.__setattr__(name, buf.float())


def _convert_jit_weights_to_fp16(traced_model, output_path, label):
    """将已 trace 但未 freeze 的 TorchScript 模型的参数/缓冲区转 FP16 后保存。

    不 freeze 直接保存，以保留参数可访问性。加载时由 _TorchScriptModelWrapper
    自行将 FP16 权重上转为 FP32 后再 freeze，保证 CPU 推理兼容。
    计算图按 FP32 trace（数值一致），磁盘存 FP16（体积减半）。
    """
    converted = 0
    for name, param in traced_model.named_parameters():
        if param.dtype == torch.float32:
            param.data = param.data.half()
            converted += 1
    for name, buf in traced_model.named_buffers():
        if buf.dtype == torch.float32:
            traced_model.__setattr__(name, buf.half())
            converted += 1
    # 不 freeze，保留参数可访问性，加载时再 freeze
    traced_model.save(output_path)
    mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"[{label}] 保存到 {output_path} ({mb:.1f} MB), 转换 {converted} 个 FP32→FP16 权重/缓冲")


def main():
    ckpt_dir = os.path.join("checkpoints", "AV_Mossformer")
    default_ckpt = os.path.join(ckpt_dir, "last_best_weights_only.pt")

    parser = argparse.ArgumentParser(description="Export stream-shaped AV-Mossformer2 TorchScript")
    parser.add_argument("--checkpoint", default=default_ckpt, help="权重文件路径")
    parser.add_argument("--fp32_out", default=os.path.join(ckpt_dir, "torch_jit.zip"), help="FP32 trace 输出")
    parser.add_argument("--script_out", default=os.path.join(ckpt_dir, "torch_jit_script_stream.zip"), help="FP32 script 输出")
    parser.add_argument("--quant_out", default=os.path.join(ckpt_dir, "torch_jit_INT8_trace_stream.zip"), help="量化 trace 输出")
    parser.add_argument("--fp16_out", default=os.path.join(ckpt_dir, "torch_jit_FP16.zip"), help="FP16 混合精度 trace 输出")
    parser.add_argument("--skip_quant", action="store_true", help="跳过量化导出")
    parser.add_argument("--skip_fp16", action="store_true", help="跳过 FP16 导出")
    parser.add_argument("--fp16_only", action="store_true", help="只做 FP16 导出（需已有 FP32 模型用于数值对比）")
    parser.add_argument("--script_mode", choices=["trace", "script", "both"], default="both")
    parser.add_argument("--chunk_ms", type=float, default=200.0, help="保留参数（不再用于 trace 形状）")
    parser.add_argument("--context_ms", type=float, default=100.0, help="流式左上下文（毫秒），与 main 一致")
    parser.add_argument(
        "--infer_chunk_ms",
        type=float,
        default=500.0,
        help="流式 hop（毫秒）；trace 窗 T_audio = context + hop + lookahead",
    )
    parser.add_argument("--lookahead_ms", type=float, default=0.0, help="与运行时 lookahead 对齐")
    parser.add_argument(
        "--fp32_fixed_out",
        default=os.path.join(ckpt_dir, "torch_jit_fixed.zip"),
        help="定长 trace 副本（与 av_mossformer2_fixed.onnx 同形，LibTorch 部署）",
    )
    parser.add_argument(
        "--export_fixed",
        action="store_true",
        help="额外将 FP32 trace 另存为 fp32_fixed_out（与 fp32_out 同图）",
    )
    parser.add_argument("--warmup_iters", type=int, default=8)
    parser.add_argument("--audio_sr", type=int, default=16000)
    parser.add_argument("--ref_sr", type=float, default=30.0)
    parser.add_argument("--image_size", type=int, default=96)
    parser.add_argument(
        "--skip_linear_names",
        default="to_qk,to_hidden,to_out",
        help="量化时跳过的 Linear 子模块名，逗号分隔",
    )
    args = parser.parse_args()

    yaml_path = os.path.join(ckpt_dir, "config.yaml")

    # ── fp16_only 模式：只做 FP16 导出，跳过 FP32/量化 ──
    if args.fp16_only:
        cfg = load_config(yaml_path)
        model = build_model(cfg, args.checkpoint)
        model.eval()
        print("[FP16-only] 模型加载完成")

        dummy_inputs = _make_dummy(
            audio_sr=int(args.audio_sr),
            ref_sr=float(args.ref_sr),
            context_ms=float(args.context_ms),
            infer_chunk_ms=float(args.infer_chunk_ms),
            lookahead_ms=float(args.lookahead_ms),
            image_size=int(args.image_size),
        )
        print(
            f"[FP16-only] stream-shape dummy: audio={tuple(dummy_inputs[0].shape)}, "
            f"ref={tuple(dummy_inputs[1].shape)}"
        )
        _warmup(model, dummy_inputs, args.warmup_iters)

        # 先计算 FP32 参考输出（在模型参数被修改之前）
        np.random.seed(42)
        t_audio_cmp, t_ref_cmp = _stream_shape(
            int(args.audio_sr),
            float(args.ref_sr),
            float(args.context_ms),
            float(args.infer_chunk_ms),
            float(args.lookahead_ms),
        )
        test_mix = torch.randn(1, t_audio_cmp, dtype=torch.float32)
        test_ref = torch.randn(1, t_ref_cmp, int(args.image_size), int(args.image_size), 3, dtype=torch.float32)
        with torch.no_grad():
            fp32_ref_out = model(test_mix, test_ref)

        # FP32 trace（不 freeze，以便后续转 FP16）
        with torch.no_grad():
            fp32_traced = torch.jit.trace(model, dummy_inputs, check_trace=False)

        # 将 FP32 trace 模型的权重转 FP16 保存（内部会 freeze）
        _convert_jit_weights_to_fp16(fp32_traced, args.fp16_out, "FP16")

        # 数值验证：加载 FP16 模型，上转 FP32 后对比输出
        fp16_loaded = torch.jit.load(args.fp16_out, map_location="cpu")
        fp16_loaded.eval()
        _upcast_jit_fp16_weights(fp16_loaded)
        fp16_loaded = torch.jit.freeze(fp16_loaded)
        with torch.no_grad():
            fp16_out = fp16_loaded(test_mix, test_ref)
        fp16_diff = (fp32_ref_out - fp16_out).abs().max().item()
        print(f"[对比] FP32 vs FP16(weight-only): max diff = {fp16_diff:.6e}")

        if os.path.isfile(args.fp16_out):
            fp16_mb = os.path.getsize(args.fp16_out) / (1024 * 1024)
            print(f"\n文件大小: FP16(weight-only)={fp16_mb:.1f} MB")
        return

    # 1) 加载模型
    cfg = load_config(yaml_path)
    model = build_model(cfg, args.checkpoint)
    model.eval()
    print("[1] 模型加载完成")

    # 2) 生成流式形态 dummy 输入 + warmup（与运行时稳态窗口完全一致）
    dummy_inputs = _make_dummy(
        audio_sr=int(args.audio_sr),
        ref_sr=float(args.ref_sr),
        context_ms=float(args.context_ms),
        infer_chunk_ms=float(args.infer_chunk_ms),
        lookahead_ms=float(args.lookahead_ms),
        image_size=int(args.image_size),
    )
    t_audio, t_ref = _stream_shape(
        int(args.audio_sr),
        float(args.ref_sr),
        float(args.context_ms),
        float(args.infer_chunk_ms),
        float(args.lookahead_ms),
    )
    print(
        f"[2] stream-shape dummy: audio=(1, {t_audio}), ref=(1, {t_ref}, {args.image_size}, "
        f"{args.image_size}, 3), warmup={int(args.warmup_iters)}"
    )
    _warmup(model, dummy_inputs, args.warmup_iters)

    fixed_also = str(args.fp32_fixed_out) if args.export_fixed else None

    # 3) 导出 FP32 TorchScript（trace/script）
    fp32_traced = None
    fp32_scripted = None
    if args.script_mode in {"trace", "both"}:
        fp32_traced = trace_and_save(
            model, dummy_inputs, args.fp32_out, "FP32-TRACE", also_save_path=fixed_also
        )
    if args.script_mode in {"script", "both"}:
        try:
            fp32_scripted = script_and_save(model, args.script_out, "FP32-SCRIPT")
        except Exception as e:
            print(f"[WARN] FP32-SCRIPT 导出失败，已跳过: {type(e).__name__}: {e}")

    # 4) 数值对比用测试输入（与稳态窗口一致，避免与 trace 形状错配）
    np.random.seed(42)
    t_audio_cmp, t_ref_cmp = _stream_shape(
        int(args.audio_sr),
        float(args.ref_sr),
        float(args.context_ms),
        float(args.infer_chunk_ms),
        float(args.lookahead_ms),
    )
    test_mix = torch.randn(1, t_audio_cmp, dtype=torch.float32)
    test_ref = torch.randn(1, t_ref_cmp, int(args.image_size), int(args.image_size), 3, dtype=torch.float32)
    with torch.no_grad():
        fp32_ref_out = model(test_mix, test_ref)
    if fp32_traced is not None:
        fp32_traced_out = fp32_traced(test_mix, test_ref)
        fp32_diff = (fp32_ref_out - fp32_traced_out).abs().max().item()
        print(f"[对比] FP32 eager vs traced: max diff = {fp32_diff:.6e}")
    if fp32_scripted is not None:
        fp32_script_out = fp32_scripted(test_mix, test_ref)
        fp32_script_diff = (fp32_ref_out - fp32_script_out).abs().max().item()
        print(f"[对比] FP32 eager vs scripted: max diff = {fp32_script_diff:.6e}")

    # 5) 导出量化 TorchScript
    if not args.skip_quant:
        # 默认跳过 attention 相关 Linear，只量化 FFN 层；可由命令行覆盖
        _SKIP_LINEAR_NAMES = _parse_skip_names(args.skip_linear_names)
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                # 取最后一层属性名判断
                leaf = name.split(".")[-1]
                if leaf in _SKIP_LINEAR_NAMES:
                    module.qconfig = None

        quant_model = torch.quantization.quantize_dynamic(
            model, {torch.nn.Linear}, dtype=torch.qint8
        )
        quant_model.eval()

        skipped = sum(1 for n, m in model.named_modules()
                      if isinstance(m, torch.nn.Linear) and n.split(".")[-1] in _SKIP_LINEAR_NAMES)
        total = sum(1 for _, m in model.named_modules() if isinstance(m, torch.nn.Linear))
        print(f"[Quant] 量化 {total - skipped}/{total} 个 Linear 层（跳过 {skipped} 个 attention 层）")
        _warmup(quant_model, dummy_inputs, max(2, int(args.warmup_iters // 2)))
        quant_traced = trace_and_save(quant_model, dummy_inputs, args.quant_out, "Quant-TRACE")

        quant_out = quant_traced(test_mix, test_ref)
        quant_diff = (fp32_ref_out - quant_out).abs().max().item()
        print(f"[对比] FP32 vs Quant: max diff = {quant_diff:.6e}")

        # 文件大小对比
        if os.path.isfile(args.fp32_out):
            fp32_mb = os.path.getsize(args.fp32_out) / (1024 * 1024)
            quant_mb = os.path.getsize(args.quant_out) / (1024 * 1024)
            print(f"\n文件大小: FP32(trace)={fp32_mb:.1f} MB, Quant(trace)={quant_mb:.1f} MB")

    # 6) 导出 FP16(weight-only) TorchScript — 权重存 FP16，运行时上转 FP32 计算
    if not args.skip_fp16:
        # 重新构建模型（避免 FP16 转换污染原始模型参数）
        cfg_fp16 = load_config(yaml_path)
        model_fp16 = build_model(cfg_fp16, args.checkpoint)
        model_fp16.eval()
        _warmup(model_fp16, dummy_inputs, max(2, int(args.warmup_iters // 2)))
        with torch.no_grad():
            fp16_traced = torch.jit.trace(model_fp16, dummy_inputs, check_trace=False)
        _convert_jit_weights_to_fp16(fp16_traced, args.fp16_out, "FP16")

        # 数值验证
        fp16_loaded = torch.jit.load(args.fp16_out, map_location="cpu")
        fp16_loaded.eval()
        _upcast_jit_fp16_weights(fp16_loaded)
        fp16_loaded = torch.jit.freeze(fp16_loaded)
        with torch.no_grad():
            fp16_out = fp16_loaded(test_mix, test_ref)
        fp16_diff = (fp32_ref_out - fp16_out).abs().max().item()
        print(f"[对比] FP32 vs FP16(weight-only): max diff = {fp16_diff:.6e}")

        # 文件大小汇总
        sizes = {}
        if os.path.isfile(args.fp32_out):
            sizes["FP32"] = os.path.getsize(args.fp32_out) / (1024 * 1024)
        if os.path.isfile(args.fp16_out):
            sizes["FP16"] = os.path.getsize(args.fp16_out) / (1024 * 1024)
        if sizes:
            print(f"\n文件大小: " + ", ".join(f"{k}={v:.1f} MB" for k, v in sizes.items()))


if __name__ == "__main__":
    main()
