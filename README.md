# AV-MossFormer 流式语音分离

基于 MossFormer 的音视频流式语音分离 SDK。

## 三层流水线

```
原始数据 (8ch PCM + MP4)
   ↓
[1. 预处理] 读文件脚本/prepare_chunk_input.py
   → a_i: (T,) float32 mono, v_i: List[(H,W,3) uint8 BGR]
   ↓
[2. 人脸门控] face_detector.py
   → 有人脸: 放行 a_i + crops  |  无人脸: 跳过不输出
   ↓
[3. 推理] stream_inference_SDK.py
   → 分离音频 wav
```

---

## 1. 预处理

[读文件脚本/prepare_chunk_input.py](读文件脚本/prepare_chunk_input.py) — 将 8 通道裸 PCM + MP4 转为标准 `a_i` / `v_i`。

- `a_i` — `(T,) float32` mono @16kHz，通道 3-6 混合
- `v_i` — `List[(H,W,3) uint8 BGR]`，原始视频帧

详见 [读文件脚本/README.md](读文件脚本/README.md)。

---

## 2. 人脸门控

[face_detector.py](face_detector.py) — MediaPipe BlazeFace 检测 + 目标跟踪。

- 有人脸 → 输出 crops + face_valid → 送入推理层
- 无人脸 → 跳过，不输出

```bash
# 单独测试人脸检测
python main.py --face-only [--files 73]
```

详见 [face_detector_README.md](face_detector_README.md)。

---

## 3. 推理

### 依赖

```bash
conda activate av_tse_infer
```

核心依赖：numpy, opencv-python, soundfile, mediapipe, onnxruntime, PyTorch（torch 后端时）

### 测试模式（命令行）

```bash
python main.py \
  --type onnx \
  --ref_onnx_path checkpoints/AV_Mossformer/av_mossformer_ref_fixed.onnx \
  --sep_onnx_path checkpoints/AV_Mossformer/av_mossformer_sep_rknn.onnx \
  --infer_chunk_ms 500 \
  --context_ms 100 \
  --chunk_ms 100 \
  --use_stream_cache 1
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--type` | `onnx` | 推理后端：torch / onnx / onnx_quant_dynamic / torch_jit / torch_jit_fp16 |
| `--infer_chunk_ms` | `500` | 推理窗口时长（ms） |
| `--chunk_ms` | `100` | 输入切片粒度（ms） |
| `--context_ms` | `100` | 左上下文时长（ms） |
| `--use_stream_cache` | `1` | 1=full-buffer 推理，0=逐 hop 滑窗 |
| `--ref_onnx_path` | `checkpoints/AV_Mossformer/av_mossformer_ref_fixed.onnx` | ref_encoder ONNX |
| `--sep_onnx_path` | `checkpoints/AV_Mossformer/av_mossformer_sep_rknn.onnx` | separator ONNX |

输入目录结构：

```
测试用例/音频/  →  .wav 文件（16kHz，单/双声道）
测试用例/视频/  →  同名 .mp4 文件
```

输出：`测试结果/{name}_out.wav`

### 流式模式（三层串联）

```python
from stream_inference_SDK import StreamInferenceSDK, StreamProcessor
from face_detector import VisualPreprocessor

# ── 第 1 层：预处理 ──
# 流式接收 100ms 原始数据，用 prepare_chunk_input.py 中的函数转换
# from 读文件脚本.prepare_chunk_input import mix_channels
# a_i = mix_channels(raw_pcm_8ch, mode="clean")   # (1600,) float32 mono
# v_i = raw_frames                                  # List[uint8 BGR]

# ── 第 2+3 层：人脸门控 + 推理引擎 ──
streamer = StreamInferenceSDK(
    ref_onnx_path="checkpoints/AV_Mossformer/av_mossformer_ref_fixed.onnx",
    sep_onnx_path="checkpoints/AV_Mossformer/av_mossformer_sep_rknn.onnx",
    infer_chunk_ms=500, context_ms=100, use_stream_cache=1,
    face_detector="none",
)
crop_size = int(streamer._core._tracker_args["crop_size"])

detector = VisualPreprocessor(
    crop_size=crop_size,
    model_path="blaze_face_full_range.tflite",
    detect_every_n=5, score_smooth_alpha=0.8, box_smooth_alpha=0,
    min_detection_confidence=0.5, use_lip_center_crop=True,
)

processor = StreamProcessor(
    streamer=streamer, preprocessor=detector,
    sr=16000, fps=24.0, infer_chunk_ms=500,
)

# ── 流式循环 ──
while streaming:
    a_i, v_i = get_next_chunk()       # 来自预处理层
    segs = processor.feed_chunk(a_i, v_i)
    for seg in segs:
        send_to_output(seg)

segs = processor.flush()
streamer.close()
```

### StreamProcessor API

| 方法 | 说明 |
|------|------|
| `__init__(streamer, preprocessor, sr, fps, infer_chunk_ms)` | 绑定推理引擎 + 人脸检测器 |
| `feed_chunk(audio, frames)` | 喂入一个 chunk，累积到 infer_chunk_ms 后自动推理 |
| `flush()` | 清空 buffer，返回尾部 seg |
| `reset()` | 重置累积 buffer、人脸 tracker、计时器 |

- `audio` — `np.ndarray`，shape `(T,)` 或 `(C,T)`，float32
- `frames` — `list[np.ndarray]`，每帧 BGR uint8 `(H,W,3)`
- 返回 — `list[np.ndarray]`，分离后音频段。无人脸窗口返回 `[]`

---

## 人脸检测参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--face_detector_model` | `blaze_face_full_range.tflite` | MediaPipe 模型路径 |
| `--detect_every_n` | `5` | 检测间隔帧数 |
| `--face_scale` | `1.0` | 人脸框缩放系数 |
| `--score_smooth_alpha` | `0.8` | 置信度 EMA 平滑 |
| `--box_smooth_alpha` | `0` | 人脸框 EMA 平滑 (0=不平滑) |
| `--min_detection_confidence` | `0.5` | 最低检测阈值 |
| `--mediapipe_detect_max_side` | `320` | 检测前降采样最大边长 |
| `--mediapipe_lip_crop` | `1` | 嘴部关键点裁剪 |
| `--mediapipe_lip_crop_scale` | `0.55` | 嘴裁边长相对人脸框比例 |
| `--area_switch_ratio` | `1.2` | 面积切换比例阈值 |
| `--area_switch_min_frames` | `30` | 面积切换最小持续帧数 |
| `--area_switch_min_confidence` | `0.6` | 面积切换置信度阈值 |
| `--annotate` | `True` | 输出标注视频 |
| `--write_lip` | 不启用 | 输出嘴唇裁剪视频（调试用） |

---

## 关键行为

- **无人脸跳过**：推理窗口内无人脸则跳过，不输出
- **人脸断连恢复**：无人脸后再有人脸，以 `is_start=True` 重置推理引擎
- **目标锁定**：首帧按面积+居中选主说话人，后续 IoU 跟踪同一人
- **目标丢失**：丢失 <1s 保持原位等恢复；丢失 >1s 重选
- **flush 尾部**：流结束剩余数据通过 `flush()` 输出

---

## ONNX 拆分部署

```bash
python main.py \
  --type onnx \
  --ref_onnx_path checkpoints/AV_Mossformer/av_mossformer_ref_fixed.onnx \
  --sep_onnx_path checkpoints/AV_Mossformer/av_mossformer_sep_rknn.onnx \
  --infer_chunk_ms 500
```

| 参数 | 说明 |
|------|------|
| `--onnx_path` | 单模型路径（不拆分时） |
| `--ref_onnx_path` | ref_encoder ONNX |
| `--sep_onnx_path` | separator ONNX |
| `--onnx_fixed` | 使用定长 ONNX |

## TorchScript

```bash
python main.py --type torch_jit --ts_path checkpoints/AV_Mossformer/torch_jit.zip
python main.py --type torch_jit --torch_jit_fixed
```
