"""Export AV_Mossformer2 TSE model to ONNX (FP32 + 静态 INT8 + 混合精度 FP16).

用法 (conda env: av_tse_infer):
    python 脚本/export_onnx.py                           # 动态输入 FP32（默认）
    python 脚本/export_onnx.py --fixed --fp32_out checkpoints/AV_Mossformer/av_mossformer2_fixed.onnx
    python 脚本/export_onnx.py --fixed --context_ms 100 --infer_chunk_ms 500 --skip_quant

定长导出需与 main.py 流式参数一致（默认 context_ms=100、infer_chunk_ms=500）：
  T_audio = context + hop + lookahead 采样点（默认 9600 @16kHz）
  T_ref   = round(T_audio / audio_sr * ref_sr)（默认 18 @30fps）

输出:
    checkpoints/AV_Mossformer/av_mossformer2.onnx              (FP32, 动态)
    checkpoints/AV_Mossformer/av_mossformer2_fixed.onnx        (FP32, 定长)
    checkpoints/AV_Mossformer/av_mossformer2_INT8.onnx         (静态 INT8)
    checkpoints/AV_Mossformer/av_mossformer2_FP16.onnx       (混合精度 FP16)
"""

import argparse
import os
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

from networks import network_wrapper


def _dict_to_ns(d):
    from types import SimpleNamespace

    if isinstance(d, dict):
        return SimpleNamespace(**{k: _dict_to_ns(v) for k, v in d.items()})
    if isinstance(d, list):
        return [_dict_to_ns(v) for v in d]
    return d


def load_config(yaml_path):
    with open(yaml_path, "r") as f:
        cfg = yaml.safe_load(f)
    ns = _dict_to_ns(cfg)
    na = ns.network_audio
    na.stream_cache_enable = 0
    na.ref_stream_cache_enable = 0
    na.stream_cache_debug = 0
    return ns


def build_model(cfg, ckpt_path):
    device = torch.device("cpu")
    cfg.device = device
    model = network_wrapper(cfg).to(device)
    model.eval()

    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt

    def _strip(name):
        while name.startswith("module."):
            name = name[len("module."):]
        return name

    if isinstance(state_dict, dict) and state_dict:
        state_dict = {_strip(k): v for k, v in state_dict.items()}
        ks = list(state_dict.keys())
        if ks and not any(k.startswith("av_skim.") or k.startswith("model.") for k in ks):
            if any(k.startswith("sep_network.") or k.startswith("ref_encoder.") for k in ks):
                state_dict = {f"model.{k}": v for k, v in state_dict.items()}

    model_sd = model.state_dict()
    for key in list(model_sd.keys()):
        bare = _strip(key)
        picked = None
        if key in state_dict and model_sd[key].shape == state_dict[key].shape:
            picked = state_dict[key]
        elif bare in state_dict and model_sd[key].shape == state_dict[bare].shape:
            picked = state_dict[bare]
        elif f"module.{key}" in state_dict and model_sd[key].shape == state_dict[f"module.{key}"].shape:
            picked = state_dict[f"module.{key}"]
        elif f"module.{bare}" in state_dict and model_sd[key].shape == state_dict[f"module.{bare}"].shape:
            picked = state_dict[f"module.{bare}"]
        if picked is not None:
            model_sd[key] = picked
    model.load_state_dict(model_sd)
    print(f"[export] loaded weights from {ckpt_path}")
    return model


class MossformerOnnxExport(nn.Module):
    """Wrap network_wrapper for full or sep-only ONNX export."""

    def __init__(self, net: nn.Module, export_part: str = "full", ref_is_gray: bool = False):
        super().__init__()
        self.net = net
        self.export_part = str(export_part)
        self.ref_is_gray = ref_is_gray  # True: ref is already 4D grayscale [B, T, H, W]

    def forward(self, mixture, ref):
        if self.export_part == "sep":
            return self._forward_sep(mixture, ref)
        # Full model: convert RGB to gray if needed
        if not self.ref_is_gray:
            ref = self.net._video_rgb_to_gray(ref)
        return self.net.model(mixture, ref)

    def _forward_sep(self, mixture, ref):
        ref = ref.to(mixture.device)
        if not self.ref_is_gray:
            ref = self.net._video_rgb_to_gray(ref)
        h = getattr(self.net.args.network_audio, "mossformer_face_size", None) or getattr(
            self.net.args.network_audio, "image_size", 112
        )
        w = h
        if ref.shape[2] != h or ref.shape[3] != w:
            b, t, _, _ = ref.shape
            ref = ref.reshape(b * t, 1, ref.shape[2], ref.shape[3])
            ref = F.interpolate(ref, size=(h, w), mode="bilinear", align_corners=False)
            ref = ref.reshape(b, t, h, w)
        mixture_w, est_mask = self.net.model.forward_sep(mixture, ref)
        return torch.cat([mixture_w, est_mask], dim=1)


class MossformerDecoderOnnxExport(nn.Module):
    """Decoder + tail pad; inputs mixture_w, est_mask, target_audio_len (scalar tensor)."""

    def __init__(self, net: nn.Module):
        super().__init__()
        self.decoder = net.model.sep_network.decoder

    def forward(self, mixture_w, est_mask, target_audio_len: torch.Tensor):
        est = self.decoder(mixture_w, est_mask)
        pad_len = target_audio_len.reshape(-1)[0] - est.size(-1)
        tail = est.new_zeros(est.size(0), pad_len)
        return torch.cat([est, tail], dim=-1)


def _configure_decoder_ola_gather_add(model, chunk_size: int = 512) -> None:
    """Use chunked const-index Add OLA (no ScatterElements in ONNX)."""
    decoder = _resolve_mossformer_decoder(model)
    if decoder is None:
        return
    decoder.set_ola_export_mode("gather_add", chunk_size=int(chunk_size))
    print(f"[export] decoder OLA mode=gather_add chunk_size={chunk_size}")


def _resolve_mossformer_decoder(model) -> Optional[nn.Module]:
    decoder = getattr(model, "decoder", None)
    if decoder is not None:
        return decoder
    inner = getattr(model, "net", None) or getattr(model, "model", None)
    if inner is None:
        return None
    if hasattr(inner, "model"):
        inner = inner.model
    sep = getattr(inner, "sep_network", None)
    if sep is None:
        return None
    return getattr(sep, "decoder", None)


def _pin_decoder_ola_for_export(model, t_audio: int, kernel_size: int = 16) -> None:
    """Set decoder OLA fold matrix for fixed-length ONNX trace (avoids 100k+ node loop unroll)."""
    decoder = _resolve_mossformer_decoder(model)
    if decoder is None:
        return
    from models.av_mossformer2_tse.av_mossformer2 import encoder_frame_count

    t_enc = encoder_frame_count(int(t_audio), int(kernel_size))
    decoder.set_fixed_ola_frames(t_enc)
    print(f"[export] decoder OLA pinned T_frames={t_enc} (audio_len={t_audio})")


def compute_stream_window_lengths(
    audio_sr: int = 16000,
    ref_sr: float = 30.0,
    context_ms: float = 100.0,
    infer_chunk_ms: float = 500.0,
    lookahead_ms: float = 0.0,
) -> tuple[int, int]:
    """与 AVStreamInference 单 hop 稳态窗一致：context + hop + lookahead。"""
    context_samples = max(0, int(round(float(audio_sr) * (float(context_ms) / 1000.0))))
    hop_samples = max(1, int(round(float(audio_sr) * (float(infer_chunk_ms) / 1000.0))))
    lookahead_samples = max(0, int(round(float(audio_sr) * (float(lookahead_ms) / 1000.0))))
    t_audio = max(256, context_samples + hop_samples + lookahead_samples)
    t_ref = max(2, int(round(float(t_audio) / float(audio_sr) * float(ref_sr))))
    return int(t_audio), int(t_ref)


def export_onnx(
    model,
    output_path,
    opset,
    audio_sr: int = 16000,
    ref_sr: float = 30.0,
    context_ms: float = 100.0,
    infer_chunk_ms: float = 500.0,
    lookahead_ms: float = 0.0,
    image_size: int = 96,
    fixed: bool = False,
    decoder_ola: str = "scatter",
    export_part: str = "full",
    ref_is_gray: bool = False,
):
    if fixed:
        t_audio, t_ref = compute_stream_window_lengths(
            audio_sr=audio_sr,
            ref_sr=ref_sr,
            context_ms=context_ms,
            infer_chunk_ms=infer_chunk_ms,
            lookahead_ms=lookahead_ms,
        )
    else:
        # 动态导出：dummy 仅用于 trace，仍用 context+lookahead 量级
        t_audio = max(
            256,
            int(round(float(audio_sr) * (float(context_ms + infer_chunk_ms + lookahead_ms) / 1000.0))),
        )
        t_ref = max(2, int(round(float(t_audio) / float(audio_sr) * float(ref_sr))))

    dummy_mixture = torch.randn(1, t_audio, dtype=torch.float32)
    if ref_is_gray:
        # 4D grayscale ref: [B, T, H, W], RKNN friendly (no 5D ops)
        # Note: rgb_to_gray should be done as preprocessing before feeding to RKNN
        dummy_ref = torch.randn(1, t_ref, int(image_size), int(image_size), dtype=torch.float32)
    else:
        # 5D RGB ref: [B, T, H, W, 3]
        dummy_ref = torch.randn(1, t_ref, int(image_size), int(image_size), 3, dtype=torch.float32)

    export_part = str(export_part)
    if export_part == "sep":
        input_names = ["mixture", "ref"]
        output_names = ["sep_pack"]
        export_inputs = (dummy_mixture, dummy_ref)
    elif export_part == "decoder":
        from models.av_mossformer2_tse.av_mossformer2 import encoder_frame_count

        t_enc = encoder_frame_count(int(t_audio), 16)
        dec = _resolve_mossformer_decoder(model)
        n_ch = int(getattr(dec, "N", 512)) if dec is not None else 512
        dummy_mw = torch.randn(1, n_ch, t_enc, dtype=torch.float32)
        dummy_mask = torch.randn(1, n_ch, t_enc, dtype=torch.float32)
        dummy_tlen = torch.tensor([int(t_audio)], dtype=torch.int64)
        input_names = ["mixture_w", "est_mask", "target_audio_len"]
        output_names = ["output"]
        export_inputs = (dummy_mw, dummy_mask, dummy_tlen)
    else:
        input_names = ["mixture", "ref"]
        output_names = ["output"]
        export_inputs = (dummy_mixture, dummy_ref)

    do_fold = True
    if export_part == "full" and decoder_ola == "conv":
        do_fold = False

    export_kw = dict(
        input_names=input_names,
        output_names=output_names,
        opset_version=opset,
        do_constant_folding=do_fold,
    )
    if not fixed:
        export_kw["dynamic_axes"] = {
            "mixture": {0: "batch", 1: "T_audio"},
            "ref": {0: "batch", 1: "T_ref"},
            "output": {0: "batch", 2: "T_audio"},
        }

    mode = "fixed" if fixed else "dynamic"
    ref_shape_str = f"(1, {t_ref}, {image_size}, {image_size})" if ref_is_gray else f"(1, {t_ref}, {image_size}, {image_size}, 3)"
    print(
        f"[ONNX/{mode}] tracing (opset={opset}) ... "
        f"mixture=(1, {t_audio}), ref={ref_shape_str}"
    )
    if export_part == "full" and fixed and decoder_ola in ("scatter", "gather_add"):
        _pin_decoder_ola_for_export(model, t_audio)
        if decoder_ola == "gather_add":
            _configure_decoder_ola_gather_add(model)
    elif export_part == "full" and fixed:
        print(f"[export] decoder OLA mode={decoder_ola} (ConvTranspose1d, no scatter pin)")
    if export_part == "decoder" and fixed and decoder_ola in ("scatter", "gather_add"):
        dec = _resolve_mossformer_decoder(model)
        if dec is not None:
            from models.av_mossformer2_tse.av_mossformer2 import encoder_frame_count

            t_enc = encoder_frame_count(int(t_audio), 16)
            dec.set_fixed_ola_frames(t_enc)
            if decoder_ola == "gather_add":
                dec.set_ola_export_mode("gather_add", chunk_size=512)
    with torch.no_grad():
        torch.onnx.export(model, export_inputs, output_path, **export_kw)
    mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"[ONNX/{mode}] 保存到 {output_path} ({mb:.1f} MB)")
    return int(t_audio), int(t_ref)


# ──────────────────────────── 静态 INT8 量化 ────────────────────────────

class AVCalibrationDataReader:
    """为静态量化提供校准数据。"""

    def __init__(self, calibration_data):
        self.data = calibration_data
        self.index = 0

    def get_next(self):
        if self.index >= len(self.data):
            return None
        data = self.data[self.index]
        self.index += 1
        return data

    def rewind(self):
        self.index = 0


def _make_calibration_data(
    wav_path=None,
    mp4_path=None,
    n_calib=200,
    audio_sr=16000,
    ref_sr=25,
    chunk_s=1.0,
    t_audio_fixed: Optional[int] = None,
    t_ref_fixed: Optional[int] = None,
    image_size: int = 96,
):
    """从真实音视频文件生成校准数据。定长 ONNX 时传入 t_audio_fixed / t_ref_fixed。"""

    if t_audio_fixed is not None and t_ref_fixed is not None:
        chunk_audio = int(t_audio_fixed)
        chunk_ref = int(t_ref_fixed)
    else:
        chunk_audio = int(audio_sr * chunk_s)
        chunk_ref = int(ref_sr * chunk_s)
    data = []

    # 尝试加载真实音频
    wav = None
    if wav_path and os.path.isfile(wav_path):
        try:
            import soundfile as sf
            wav_file, sr = sf.read(wav_path, dtype="float32", always_2d=True)
            wav = wav_file.T  # (C, T)
            if wav.shape[0] > 1:
                wav = wav.mean(axis=0, keepdims=True)
            print(f"[Calib] 加载音频: {wav_path}, shape={wav.shape}, sr={sr}")
        except Exception as e:
            print(f"[Calib] 音频加载失败: {e}")

    # 尝试加载真实视频帧
    frames = None
    fps = ref_sr
    if mp4_path and os.path.isfile(mp4_path):
        try:
            import cv2
            cap = cv2.VideoCapture(mp4_path)
            fps = float(cap.get(cv2.CAP_PROP_FPS))
            if not np.isfinite(fps) or fps <= 1e-3:
                fps = 25.0
            frames = []
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                frames.append(frame)
            cap.release()
            # 归一化: BGR uint8 → RGB float32, 与模型预处理一致
            frames_norm = []
            for f in frames:
                f_rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
                f_resized = cv2.resize(f_rgb, (image_size, image_size), interpolation=cv2.INTER_AREA)
                f_norm = (f_resized - 0.506362) / 0.272877
                frames_norm.append(f_norm)

            # 重采样: 源 fps → 目标 ref_sr (25fps)
            src_len = len(frames_norm)
            if fps != ref_sr and src_len > 0:
                duration_s = src_len / fps
                tgt_len = max(1, int(round(duration_s * ref_sr)))
                resampled = []
                for ti in range(tgt_len):
                    si = int(np.clip(round((ti / ref_sr) * fps), 0, src_len - 1))
                    resampled.append(frames_norm[si])
                frames_norm = resampled
                print(f"[Calib] 重采样 {src_len}帧@{fps:.0f}fps → {tgt_len}帧@{ref_sr}fps")

            print(f"[Calib] 加载视频: {mp4_path}, {len(frames_norm)} 帧@{ref_sr}fps")
            frames = frames_norm
        except Exception as e:
            print(f"[Calib] 视频加载失败: {e}")

    # 从真实数据切片
    if wav is not None and frames is not None:
        n_audio_chunks = wav.shape[1] // chunk_audio
        n_video_chunks = len(frames) // chunk_ref
        n_chunks = min(n_audio_chunks, n_video_chunks)
        print(f"[Calib] 可切出 {n_chunks} 个 chunk (音频 {n_audio_chunks}, 视频 {n_video_chunks})")
        for i in range(min(n_chunks, n_calib)):
            a_start = i * chunk_audio
            mix = wav[0, a_start:a_start + chunk_audio][np.newaxis, :]
            v_start = i * chunk_ref
            ref = np.stack(frames[v_start:v_start + chunk_ref], axis=0)[np.newaxis, :]
            data.append({"mixture": mix, "ref": ref})
    else:
        # 退化: 随机数据
        print(f"[Calib] 使用随机校准数据 ({n_calib} 条)")
        for _ in range(n_calib):
            mix = np.random.randn(1, chunk_audio).astype(np.float32) * 0.1
            ref = (
                np.random.randn(1, chunk_ref, image_size, image_size, 3).astype(np.float32) * 0.1
            )
            data.append({"mixture": mix, "ref": ref})

    print(f"[Calib] 生成 {len(data)} 条校准数据")
    return data


def quantize_static_onnx(
    fp32_path,
    quant_path,
    n_calib=200,
    calib_wav=None,
    calib_mp4=None,
    t_audio_fixed: Optional[int] = None,
    t_ref_fixed: Optional[int] = None,
    image_size: int = 96,
):
    """ONNX Runtime 静态 INT8 量化。"""
    try:
        import onnx
        from onnxruntime.quantization import (
            quantize_static, QuantType, QuantFormat,
            CalibrationDataReader,
        )
    except ImportError:
        print("[Quant-Static] 跳过: onnxruntime.quantization 不可用")
        return False

    # 排除 attention 相关节点
    _SKIP_PATTERNS = ["to_qk", "to_hidden", "to_out", "to_u", "to_v"]
    onnx_model = onnx.load(fp32_path)
    nodes_to_exclude = []
    for node in onnx_model.graph.node:
        if any(p in node.name for p in _SKIP_PATTERNS):
            nodes_to_exclude.append(node.name)
    print(f"[Quant-Static] 排除 {len(nodes_to_exclude)} 个 attention 节点")

    # 校准数据
    calib_data = _make_calibration_data(
        wav_path=calib_wav,
        mp4_path=calib_mp4,
        n_calib=n_calib,
        t_audio_fixed=t_audio_fixed,
        t_ref_fixed=t_ref_fixed,
        image_size=image_size,
    )
    data_reader = AVCalibrationDataReader(calib_data)

    print(f"[Quant-Static] 量化 {fp32_path} -> {quant_path} ...")
    quantize_static(
        fp32_path,
        quant_path,
        calibration_data_reader=data_reader,
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QUInt8,
        op_types_to_quantize=["MatMul", "Conv"],
        nodes_to_exclude=nodes_to_exclude,
        per_channel=False,
        reduce_range=False,
    )
    mb = os.path.getsize(quant_path) / (1024 * 1024)
    print(f"[Quant-Static] 保存到 {quant_path} ({mb:.1f} MB)")
    return True


# ──────────────────────────── 混合精度 FP16 ────────────────────────────

def convert_mixed_fp16(fp32_path, fp16_path):
    """混合精度 FP16：权重转 FP16，敏感算子保留 FP32。"""
    try:
        import onnx
        from onnx import numpy_helper, TensorProto
    except ImportError:
        print("[FP16] 跳过: onnx 包不可用")
        return False

    # 敏感算子保留 FP32（Softmax、Sigmoid、InstanceNorm、ReduceMean 等精度敏感）
    _FP32_OPS = {"Softmax", "Sigmoid", "InstanceNormalization", "ReduceMean",
                 "LogSoftmax", "LayerNormalization"}

    onnx_model = onnx.load(fp32_path)

    # 1) 初始化器（权重）转 FP16
    fp16_initializers = set()
    for init in onnx_model.graph.initializer:
        if init.data_type == TensorProto.FLOAT:
            w = numpy_helper.to_array(init)
            w_fp16 = w.astype(np.float16)
            new_init = numpy_helper.from_array(w_fp16, name=init.name)
            init.CopyFrom(new_init)
            fp16_initializers.add(init.name)

    # 2) 对敏感算子的输入/输出插入 Cast 节点保持 FP32
    cast_count = 0
    for node in onnx_model.graph.node:
        if node.op_type not in _FP32_OPS:
            continue
        # 输入: FP16 → FP32
        for i, inp_name in enumerate(node.input):
            if inp_name in fp16_initializers:
                continue  # 初始化器会自动 cast
            cast_name = f"{node.name}_fp16_cast_in_{i}"
            # 在 node 前插入 Cast FP16→FP32
            cast_node = onnx.helper.make_node(
                "Cast", inputs=[inp_name], outputs=[cast_name],
                name=cast_name, to=TensorProto.FLOAT,
            )
            onnx_model.graph.node.insert(
                list(onnx_model.graph.node).index(node), cast_node
            )
            node.input[i] = cast_name
            cast_count += 1
        # 输出: FP32 → FP16
        for i, out_name in enumerate(node.output):
            cast_name = f"{node.name}_fp16_cast_out_{i}"
            cast_node = onnx.helper.make_node(
                "Cast", inputs=[out_name], outputs=[cast_name],
                name=cast_name, to=TensorProto.FLOAT16,
            )
            # 在 node 后插入 Cast FP32→FP16
            idx = list(onnx_model.graph.node).index(node)
            onnx_model.graph.node.insert(idx + 1, cast_node)
            # 更新后续节点的输入引用
            for later_node in onnx_model.graph.node:
                for j, later_inp in enumerate(later_node.input):
                    if later_inp == out_name:
                        later_node.input[j] = cast_name
            cast_count += 1

    # 3) 更新模型输入输出类型
    for inp in onnx_model.graph.input:
        if inp.type.tensor_type.elem_type == TensorProto.FLOAT:
            inp.type.tensor_type.elem_type = TensorProto.FLOAT16
    for out in onnx_model.graph.output:
        # 保持输出为 FP32（方便下游使用）
        pass

    onnx.save(onnx_model, fp16_path)
    mb = os.path.getsize(fp16_path) / (1024 * 1024)
    print(f"[FP16] 保存到 {fp16_path} ({mb:.1f} MB), 插入 {cast_count} 个 Cast 节点")
    return True


# ──────────────────────────── 动态 INT8 量化 ────────────────────────────

def quantize_dynamic_onnx(input_path, quant_path):
    """动态 INT8 量化：只量化 MatMul，跳过精度敏感的 attention/output/decoder 节点。

    可对 FP32 或 FP16 ONNX 模型使用，配合 convert_mixed_fp16 实现 FP16+INT8 组合。
    """
    try:
        import onnx
        from onnxruntime.quantization import QuantType, quantize_dynamic
    except ImportError:
        print("[Quant-Dynamic] 跳过: onnx/onnxruntime 不可用")
        return False

    onnx_model = onnx.load(input_path)
    nodes = [n.name for n in onnx_model.graph.node]
    _SKIP_PATTERNS = ["to_qk", "to_hidden", "to_out", "to_u", "to_v", "output", "decoder"]
    nodes_to_exclude = [m for m in nodes if any(p in m for p in _SKIP_PATTERNS)]

    print(f"[Quant-Dynamic] 排除 {len(nodes_to_exclude)} 个敏感节点，"
          f"量化其余 MatMul (per_channel, QUInt8)")
    quantize_dynamic(
        model_input=input_path,
        model_output=quant_path,
        op_types_to_quantize=["MatMul"],
        per_channel=True,
        reduce_range=False,
        weight_type=QuantType.QUInt8,
        nodes_to_exclude=nodes_to_exclude,
    )
    mb = os.path.getsize(quant_path) / (1024 * 1024)
    print(f"[Quant-Dynamic] 保存到 {quant_path} ({mb:.1f} MB)")
    return True


# ──────────────────────────── 验证 ────────────────────────────

def verify(
    onnx_path,
    model=None,
    label="ONNX",
    t_audio: Optional[int] = None,
    t_ref: Optional[int] = None,
    image_size: int = 96,
):
    try:
        import onnx
        import onnxruntime as ort
    except ImportError:
        print(f"[{label}] 跳过验证: onnx/onnxruntime 未安装")
        return

    try:
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model, full_check=False)
        print(f"[{label}] ONNX 模型校验通过")
    except Exception as e:
        print(f"[{label}] ONNX 模型校验失败: {e}")

    try:
        sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    except Exception as e:
        print(f"[{label}] 加载失败: {e}")
        return

    inp = sess.get_inputs()
    out = sess.get_outputs()
    print(f"[{label}] inputs:  {[(i.name, i.shape, i.type) for i in inp]}")
    print(f"[{label}] outputs: {[(o.name, o.shape, o.type) for o in out]}")

    if t_audio is None or t_ref is None:
        t_audio = 64000
        t_ref = 100
    np.random.seed(42)
    mix_np = np.random.randn(1, int(t_audio)).astype(np.float32)
    # Auto-detect ref shape based on ONNX model input (5D RGB or 4D grayscale)
    ref_shape = None
    for i in inp:
        if i.name == "ref":
            ref_shape = i.shape
            break
    if ref_shape is not None and len(ref_shape) == 4:
        # 4D grayscale ref: [B, T, H, W]
        ref_np = np.random.randn(1, int(t_ref), image_size, image_size).astype(np.float32)
    else:
        # 5D RGB ref: [B, T, H, W, 3]
        ref_np = np.random.randn(1, int(t_ref), image_size, image_size, 3).astype(np.float32)

    try:
        result = sess.run(None, {"mixture": mix_np, "ref": ref_np})
        print(f"[{label}] inference OK, output shape: {result[0].shape}")
    except Exception as e:
        print(f"[{label}] inference FAILED: {e}")
        return

    if model is not None:
        with torch.no_grad():
            pt_out = model(torch.from_numpy(mix_np), torch.from_numpy(ref_np)).numpy()
        diff = np.abs(pt_out - result[0]).max()
        print(f"[{label}] max|PyTorch - ONNX| = {diff:.6e}")


# ──────────────────────────── 主流程 ────────────────────────────

def main():
    ckpt_dir = os.path.join("checkpoints", "AV_Mossformer")
    default_ckpt = os.path.join(ckpt_dir, "last_best_weights_only.pt")
    default_fp32 = os.path.join(ckpt_dir, "av_mossformer2.onnx")
    default_fp32_fixed = os.path.join(ckpt_dir, "av_mossformer2_fixed.onnx")
    default_quant = os.path.join(ckpt_dir, "av_mossformer2_INT8.onnx")
    default_fp16 = os.path.join(ckpt_dir, "av_mossformer2_FP16.onnx")
    default_fp16_int8 = os.path.join(ckpt_dir, "av_mossformer2_FP16_INT8.onnx")

    parser = argparse.ArgumentParser(description="Export AV_Mossformer2 to ONNX (FP32 + INT8 + FP16 + FP16+INT8)")
    parser.add_argument(
        "--fixed",
        action="store_true",
        help="导出定长输入 ONNX（与 --context_ms/--infer_chunk_ms 对齐 main.py 流式窗）",
    )
    parser.add_argument("--checkpoint", default=default_ckpt, help="权重文件路径")
    parser.add_argument("--fp32_out", default=default_fp32, help="FP32 ONNX 输出路径")
    parser.add_argument("--quant_out", default=default_quant, help="静态 INT8 ONNX 输出路径")
    parser.add_argument("--fp16_out", default=default_fp16, help="FP16 ONNX 输出路径")
    parser.add_argument("--fp16_int8_out", default=default_fp16_int8, help="FP16+INT8 ONNX 输出路径")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset (默认 17)")
    parser.add_argument("--audio_sr", type=int, default=16000, help="导出 dummy 的音频采样率")
    parser.add_argument("--ref_sr", type=float, default=30.0, help="导出 dummy 的视频帧率")
    parser.add_argument("--context_ms", type=float, default=100.0, help="流式左上下文（毫秒），定长默认与 main 一致")
    parser.add_argument(
        "--infer_chunk_ms",
        type=float,
        default=500.0,
        help="流式 hop 时长（毫秒），定长 T_audio = context + hop + lookahead",
    )
    parser.add_argument("--lookahead_ms", type=float, default=0.0, help="流式前瞻窗口（毫秒）")
    parser.add_argument("--image_size", type=int, default=96, help="导出 dummy 的人脸尺寸")
    parser.add_argument(
        "--decoder_ola",
        choices=("scatter", "conv", "gather_add"),
        default="scatter",
        help="decoder overlap-add: scatter (RKNN may fail), conv (ConvTranspose1d), gather_add (chunked Add)",
    )
    parser.add_argument(
        "--export_part",
        choices=("full", "sep", "decoder"),
        default="full",
        help="full=端到端; sep=encoder+separator(RKNN); decoder=仅 decoder(ORT/CPU)",
    )
    parser.add_argument(
        "--ref_is_gray",
        action="store_true",
        help="ref 输入已经是 4D 灰度图 [B, T, H, W]，不需要在模型内部转灰度（用于 RKNN 部署，避免 5D op）",
    )
    parser.add_argument("--skip_verify", action="store_true", help="跳过验证")
    parser.add_argument("--skip_quant", action="store_true", help="跳过静态 INT8 量化")
    parser.add_argument("--skip_fp16", action="store_true", help="跳过 FP16 转换")
    parser.add_argument("--skip_fp16_int8", action="store_true", help="跳过 FP16+INT8 导出")
    parser.add_argument("--quant_only", action="store_true", help="只做静态量化，跳过 FP32 导出")
    parser.add_argument("--fp16_only", action="store_true", help="只做 FP16 转换，跳过 FP32 导出")
    parser.add_argument("--fp16_int8_only", action="store_true", help="只做 FP16+INT8 导出（需已有 FP16 ONNX）")
    parser.add_argument("--n_calib", type=int, default=200, help="静态量化校准样本数 (默认 200)")
    parser.add_argument("--calib_wav", default=None, help="校准用音频文件路径")
    parser.add_argument("--calib_mp4", default=None, help="校准用视频文件路径")
    args = parser.parse_args()
    if args.fixed and args.fp32_out == default_fp32:
        args.fp32_out = default_fp32_fixed

    model = None
    t_audio_fixed = None
    t_ref_fixed = None
    if args.fixed:
        t_audio_fixed, t_ref_fixed = compute_stream_window_lengths(
            audio_sr=int(args.audio_sr),
            ref_sr=float(args.ref_sr),
            context_ms=float(args.context_ms),
            infer_chunk_ms=float(args.infer_chunk_ms),
            lookahead_ms=float(args.lookahead_ms),
        )
        print(f"[fixed] T_audio={t_audio_fixed}, T_ref={t_ref_fixed}")

    if args.quant_only:
        if not os.path.isfile(args.fp32_out):
            print(f"[error] FP32 ONNX 不存在: {args.fp32_out}")
            return
        ok = quantize_static_onnx(
            args.fp32_out,
            args.quant_out,
            n_calib=args.n_calib,
            calib_wav=args.calib_wav,
            calib_mp4=args.calib_mp4,
            t_audio_fixed=t_audio_fixed,
            t_ref_fixed=t_ref_fixed,
            image_size=int(args.image_size),
        )
        if ok and not args.skip_verify:
            verify(
                args.quant_out,
                label="INT8-Static",
                t_audio=t_audio_fixed,
                t_ref=t_ref_fixed,
                image_size=int(args.image_size),
            )

    elif args.fp16_only:
        if not os.path.isfile(args.fp32_out):
            print(f"[error] FP32 ONNX 不存在: {args.fp32_out}")
            return
        ok = convert_mixed_fp16(args.fp32_out, args.fp16_out)
        if ok and not args.skip_verify:
            verify(
                args.fp16_out,
                label="FP16-Mixed",
                t_audio=t_audio_fixed,
                t_ref=t_ref_fixed,
                image_size=int(args.image_size),
            )

    elif args.fp16_int8_only:
        # 只做 FP16+INT8：需已有 FP16 ONNX
        fp16_path = args.fp16_out
        if not os.path.isfile(fp16_path):
            # 尝试从 FP32 先转 FP16
            if os.path.isfile(args.fp32_out):
                print(f"[FP16+INT8] FP16 ONNX 不存在，从 FP32 转换...")
                convert_mixed_fp16(args.fp32_out, fp16_path)
            else:
                print(f"[error] FP16 和 FP32 ONNX 均不存在，请先导出 FP32 ONNX")
                return
        ok = quantize_dynamic_onnx(fp16_path, args.fp16_int8_out)
        if ok and not args.skip_verify:
            verify(
                args.fp16_int8_out,
                label="FP16+INT8",
                t_audio=t_audio_fixed,
                t_ref=t_ref_fixed,
                image_size=int(args.image_size),
            )

    else:
        yaml_path = os.path.join(os.path.dirname(args.checkpoint), "config.yaml")
        if not os.path.isfile(yaml_path):
            yaml_path = os.path.join(ckpt_dir, "config.yaml")

        # 1) 加载模型
        cfg = load_config(yaml_path)
        base_model = build_model(cfg, args.checkpoint)
        export_part = str(args.export_part)
        ref_is_gray = bool(getattr(args, "ref_is_gray", False))
        if export_part == "sep":
            model = MossformerOnnxExport(base_model, export_part="sep", ref_is_gray=ref_is_gray)
        elif export_part == "decoder":
            model = MossformerDecoderOnnxExport(base_model)
        else:
            model = MossformerOnnxExport(base_model, export_part="full", ref_is_gray=ref_is_gray)

        # 2) 导出 FP32 ONNX
        t_audio_fixed, t_ref_fixed = export_onnx(
            model,
            args.fp32_out,
            args.opset,
            audio_sr=int(args.audio_sr),
            ref_sr=float(args.ref_sr),
            context_ms=float(args.context_ms),
            infer_chunk_ms=float(args.infer_chunk_ms),
            lookahead_ms=float(args.lookahead_ms),
            image_size=int(args.image_size),
            fixed=bool(args.fixed),
            decoder_ola=str(args.decoder_ola),
            export_part=export_part,
            ref_is_gray=ref_is_gray,
        )
        if not args.skip_verify:
            verify(
                args.fp32_out,
                model,
                label="FP32-Fixed" if args.fixed else "FP32",
                t_audio=t_audio_fixed if args.fixed else None,
                t_ref=t_ref_fixed if args.fixed else None,
                image_size=int(args.image_size),
            )

        # 3) 静态 INT8 量化
        if not args.skip_quant:
            ok = quantize_static_onnx(
                args.fp32_out,
                args.quant_out,
                n_calib=args.n_calib,
                calib_wav=args.calib_wav,
                calib_mp4=args.calib_mp4,
                t_audio_fixed=t_audio_fixed if args.fixed else None,
                t_ref_fixed=t_ref_fixed if args.fixed else None,
                image_size=int(args.image_size),
            )
            if ok and not args.skip_verify:
                verify(
                    args.quant_out,
                    model,
                    label="INT8-Static",
                    t_audio=t_audio_fixed if args.fixed else None,
                    t_ref=t_ref_fixed if args.fixed else None,
                    image_size=int(args.image_size),
                )

        # 4) 混合精度 FP16
        fp16_available = False
        if not args.skip_fp16:
            ok = convert_mixed_fp16(args.fp32_out, args.fp16_out)
            fp16_available = ok
            if ok and not args.skip_verify:
                verify(
                    args.fp16_out,
                    model,
                    label="FP16-Mixed",
                    t_audio=t_audio_fixed if args.fixed else None,
                    t_ref=t_ref_fixed if args.fixed else None,
                    image_size=int(args.image_size),
                )

        # 5) FP16+INT8：先 FP16 再 INT8 动态量化
        if not args.skip_fp16_int8:
            if not fp16_available and os.path.isfile(args.fp16_out):
                fp16_available = True
            if not fp16_available:
                print("[WARN] 跳过 FP16+INT8: FP16 ONNX 不可用")
            else:
                ok = quantize_dynamic_onnx(args.fp16_out, args.fp16_int8_out)
                if ok and not args.skip_verify:
                    verify(
                        args.fp16_int8_out,
                        model,
                        label="FP16+INT8",
                        t_audio=t_audio_fixed if args.fixed else None,
                        t_ref=t_ref_fixed if args.fixed else None,
                        image_size=int(args.image_size),
                    )

    # 文件大小汇总
    sizes = {}
    for label, path in [("FP32", args.fp32_out), ("INT8-Static", args.quant_out),
                        ("FP16", args.fp16_out), ("FP16+INT8", args.fp16_int8_out)]:
        if os.path.isfile(path):
            sizes[label] = os.path.getsize(path) / (1024 * 1024)
    if sizes:
        print(f"\n文件大小: " + ", ".join(f"{k}={v:.1f} MB" for k, v in sizes.items()))


if __name__ == "__main__":
    main()
