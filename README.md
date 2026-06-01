# AV-MossFormer 流式语音分离

基于 MossFormer 的音视频流式语音分离 SDK。支持 MediaPipe 人脸检测 + go/no-go 机制——无人脸的音频段自动跳过推理。

## 依赖

```bash
conda activate av_tse_infer
```

核心依赖：numpy, opencv-python, soundfile, mediapipe, onnxruntime, PyTorch（torch 后端时）

---

## 两种使用模式

### 测试模式（命令行）

从文件读取整段音视频，按 `chunk_ms` 切片模拟流式输入：

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

输入目录结构：

```
测试用例/音频/  →  .wav 文件（16kHz，单/双声道）
测试用例/视频/  →  同名 .mp4 文件
```

输出：`测试结果/{name}_out.wav`

#### 主要参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--type` | `torch` | 推理后端：torch / onnx / onnx_quant_dynamic / torch_jit / torch_jit_fp16 |
| `--infer_chunk_ms` | `200` | 推理窗口时长（ms），累积到此长度后触发推理 |
| `--chunk_ms` | `100` | 输入切片粒度（ms），模拟流式 chunk 大小 |
| `--context_ms` | `100` | 左上下文时长（ms） |
| `--use_stream_cache` | `1` | 1=full-buffer 推理，0=逐 hop 滑窗 |
| `--ref_onnx_path` | — | 拆分部署：ref_encoder ONNX 路径 |
| `--sep_onnx_path` | — | 拆分部署：separator ONNX 路径 |
| `--face_target_policy` | `center_largest_lock` | 多人脸选目标策略 |
| `--mediapipe_lip_crop` | `1` | 启用嘴部关键点裁剪 |

---

### 流式模式（Python API）

外部代码直接调用 `StreamProcessor`，逐 chunk 喂入实时数据：

```python
from stream_inference_SDK import StreamInferenceSDK, StreamProcessor
from visual_preprocessor import VisualPreprocessor

# 1) 初始化 SDK
streamer = StreamInferenceSDK(
    ref_onnx_path="checkpoints/AV_Mossformer/av_mossformer_ref_fixed.onnx",
    sep_onnx_path="checkpoints/AV_Mossformer/av_mossformer_sep_rknn.onnx",
    infer_chunk_ms=500,
    context_ms=100,
    use_stream_cache=1,
    face_detector="none",          # 必须：SDK 不自己做检测
)
crop_size = int(streamer._core._tracker_args["crop_size"])

# 2) 初始化视觉预处理
preprocessor = VisualPreprocessor(
    crop_size=crop_size,
    model_path="detector.tflite",
    use_lip_center_crop=True,
    target_policy="center_largest_lock",
    target_lock=True,
)

# 3) 初始化流式处理器
processor = StreamProcessor(
    streamer=streamer,
    preprocessor=preprocessor,
    sr=16000,
    fps=25.0,
    infer_chunk_ms=500,
)

# 4) 流式循环（每个 chunk 约 100ms 的数据）
while streaming:
    audio_chunk = get_audio_chunk()      # np.ndarray, (T,) 或 (C,T), float32
    video_frames = get_video_frames()    # list[np.ndarray], BGR uint8 (H,W,3)

    segs = processor.feed_chunk(audio_chunk, video_frames)
    for seg in segs:                     # 每个 seg 是 (T,) float32
        send_to_output(seg)

# 5) 流结束：清空 buffer
segs = processor.flush()
for seg in segs:
    send_to_output(seg)

streamer.close()
```

多路流可复用同一 `StreamProcessor`，每路开始前调用 `processor.reset()` 清空累积状态。

---

## API 参考

### StreamProcessor

```python
class StreamProcessor:
    def __init__(self, streamer, preprocessor, sr: int, fps: float, infer_chunk_ms: float)
    def feed_chunk(self, audio_chunk: np.ndarray, video_frames: list) -> list
    def flush(self) -> list
    def reset(self) -> None
```

| 方法 | 说明 |
|------|------|
| `__init__` | 绑定 SDK、预处理器、采样率、帧率、推理窗长 |
| `feed_chunk(audio, frames)` | 喂入一个 chunk。累积到 infer_chunk_ms 后自动检测+推理。返回 seg 列表 |
| `flush()` | 清空 buffer（is_end=True），返回尾部 seg。同时重置状态 |
| `reset()` | 重置累积 buffer、人脸 tracker、计时器 |

**feed_chunk 参数：**
- `audio_chunk` — `np.ndarray`，shape `(T,)` 或 `(C,T)`，float32
- `video_frames` — `list[np.ndarray]`，每帧 BGR uint8 `(H,W,3)`
- 返回 — `list[np.ndarray]`，分离后的音频段。当前窗口无人脸返回 `[]`

---

## 关键注意事项

- **无人脸自动跳过**：累积到 `infer_chunk_ms` 后，若该窗口无人脸则丢弃，不触发推理
- **人脸断连恢复**：连续无人脸后再有人脸，自动以 `is_start=True` 重置推理引擎
- **flush 尾部**：流结束时剩余不足 `infer_chunk_ms` 的数据通过 `flush()` 输出
- **输出对齐**：每个 seg 的音频样本数 = `infer_chunk_ms` 对应的采样数

---

## ONNX 拆分部署

支持 ref_encoder + separator 分别用 ONNX 推理（RKNN 拆分部署）：

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
| `--ref_onnx_path` | ref_encoder ONNX（与 sep_onnx_path 同时使用） |
| `--sep_onnx_path` | separator ONNX（与 ref_onnx_path 同时使用） |
| `--onnx_fixed` | 使用预置定长 ONNX（av_mossformer2_fixed.onnx） |

---

## TorchScript

```bash
python main.py --type torch_jit --ts_path checkpoints/AV_Mossformer/torch_jit.zip
python main.py --type torch_jit --torch_jit_fixed  # 定长 trace
```

---

## 视觉预处理参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--face_detector_model` | `detector.tflite` | MediaPipe 模型路径 |
| `--mediapipe_lip_crop` | `1` | 嘴部关键点裁剪（0=人脸框裁剪） |
| `--mediapipe_lip_crop_scale` | `0.8` | 嘴裁边长相对人脸框比例 |
| `--mediapipe_lip_crop_min_px` | `48` | 嘴裁最小边长（像素） |
| `--mediapipe_lip_crop_max_px` | `2048` | 嘴裁最大边长（像素） |
| `--face_target_policy` | `center_largest_lock` | 多人脸选取策略 |
| `--face_target_lock` | `1` | 首帧选人后按 IoU 锁定 |
| `--face_target_lock_min_iou` | `0.15` | 锁定最低 IoU 阈值 |
| `--write_lip` | 不启用 | 输出裁剪后的嘴唇视频（调试用） |
| `--write_lip_dir` | `./测试结果_嘴唇视频` | 嘴唇视频输出目录 |
