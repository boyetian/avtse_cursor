# AV_TSE（SDK 极简用法）

## 1. 流式推理：单 chunk 反复调用方式

初始化只需执行一次：

```python
from stream_inference_SDK import StreamInferenceSDK

# PyTorch 推理（默认，RTF ~3.0）
streamer = StreamInferenceSDK()

# ONNX 推理（RTF ~4.x，需先导出 onnx 模型）
# streamer = StreamInferenceSDK(onnx_path="checkpoints/AV_Mossformer/av_mossformer2.onnx")          # 动态 T_audio/T_ref
# streamer = StreamInferenceSDK(onnx_path="checkpoints/AV_Mossformer/av_mossformer2_fixed.onnx")  # 定长，推荐部署

# TorchScript 推理（RTF ~3.x，可用于 LibTorch C++ 部署）
# streamer = StreamInferenceSDK(ts_path="checkpoints/AV_Mossformer/torch_jit.zip")
# streamer = StreamInferenceSDK(ts_path="checkpoints/AV_Mossformer/torch_jit_fixed.zip")  # 定长 trace，C++ 推荐
```

每次收到一包对齐的音视频 chunk（chunk 时长可为 20/50/200/500ms 等任意值），重复调用一次 `process_av_stream`：

```python
outputs = streamer.process_av_stream(
    audio_chunk=a_chunk,      # np.ndarray, shape=(C,T), float32
    video_chunk=v_chunk,      # List[np.ndarray], 单帧 (H,W,3), uint8, BGR
    is_start=is_first_chunk,  # 第一包 True
    is_end=is_last_chunk,     # 最后一包 True
    sampling_rate=16000,
    fps=25.0,
)
```

- `outputs` 类型为 `List[np.ndarray]`：一次调用可能返回 0~N 段输出音频。  
  - 若本次输入尚未凑够 200ms（或累计后仍不足），通常返回空列表。  
  - 若本次输入超过 200ms，或累计凑够了 200ms，可能返回 1 段或多段（SDK 内部按 200ms 自动切块推理）。  
- 最后一包必须设置 `is_end=True`：会触发尾包处理，把缓存里不足 200ms 的尾巴再推一次（按当前尾包策略）。

### 1.1 输入输出约定

- **音频输入**：`a_chunk: np.ndarray`，shape `(C,T)` 或 `(T,)`，dtype `float32`；SDK 内部会按 `axis=0` 做 mean 混成单通道。
- **视频输入**：`v_chunk: List[np.ndarray]`，每帧为 `BGR uint8 (H,W,3)`。
- **输出**：`List[np.ndarray]` 的单通道 float32 片段。

### 1.2 单次调用的 chunk 格式

| 字段 | 变量名 | 类型 | shape | dtype | 说明 |
|---|---|---|---|---|---|
| 音频 chunk | `a_chunk` / `audio_chunk` | `np.ndarray` | `(C,T)` | `float32` | `C` 为通道数，`T` 为采样点数；SDK 内部会对 `C` 做 mean 混成单通道 |
| 视频 chunk | `v_chunk` / `video_chunk` | `List[np.ndarray]` | 每帧 `(H,W,3)` | 通常 `uint8` | OpenCV BGR 帧序列；200ms@25fps 通常约 5 帧（最后一包可能不足） |

## 2. 最小可跑示例（直接运行 main.py）

仓库内提供了一个最小示例 `main.py`，会：

- 读取测试用例音频/视频
- 转成 numpy（音频 `wav` 为 `(C,T)`，视频为 `frames: List[np.ndarray(H,W,3)]`）
- 调用 `streamer.process_av_stream(...)` 分块推理
- 输出 `./测试结果/stream_sdk_main.wav`

运行：

```bash
conda create -n av_tse_infer python=3.9
pip install -r requirements-infer-only.txt
conda run -n av_tse_infer python main.py               # PyTorch（默认）
conda run -n av_tse_infer python main.py --type onnx    # ONNX
conda run -n av_tse_infer python main.py --type torch_jit  # TorchScript
```

## 3. MediaPipe 人脸检测（`--face_detector mediapipe`）

流式推理从视频里裁出**参考人脸画面**再送入 TSE。默认使用 **MediaPipe FaceDetector**（`detector.tflite`），也可改回 OpenCV Haar。

### 3.1 依赖与模型

安装推理依赖即可（已包含 `mediapipe`）：

```bash
pip install -r requirements-infer-only.txt
```

- 模型：仓库根目录 `detector.tflite`（`--face_detector_model` 可改路径）。
- 单张图调试（框 + 关键点）：`facedetection.py`。

### 3.2 两种参考裁剪

| 模式 | 开关 | 送入模型的画面 |
|------|------|----------------|
| 人脸框裁剪 | `--mediapipe_lip_crop 0` | 以平滑后的**人脸正方形框** `last_box` 裁剪 |
| 嘴中心裁剪 | `--mediapipe_lip_crop 1`（`main.py` 默认 **1**） | 以 BlazeFace **嘴关键点**（默认 index **3**）为中心裁剪 |

嘴裁时，原图上的正方形边长：

`side_px = clamp(mediapipe_lip_crop_scale × max(last_box宽, last_box高), min_px, max_px)`

再缩放到 `face_crop_size`（默认 **96**，在 `av_stream_inference.py` 的 `AVStreamInference` 中配置，`main.py` 无对应 CLI）。**Overlay 绿框**仍按 `last_box` 绘制，与嘴裁范围无关。

无人脸时：不画目标框，对应时段分离输出为静音。

### 3.2.1 多人脸时选谁（`--face_target_policy`）

检测到多张脸时，**绿框 = 送入 TSE 的目标**，**黄框 = 干扰**。默认 `center_largest` + **`--face_target_lock 1`（默认开启）**：

1. **首帧**：在面积 ≥ 最大脸 **85%** 的候选里，选框中心最靠近画面中心的人；
2. **之后**：在每次重检测时，在与上一目标框 **IoU 最大** 的人脸间跟踪，避免 4s/20s 等处因两人大小接近而左右跳框。

| 策略 | 说明 |
|------|------|
| `center_largest`（默认） | 大面积里选最居中；配合锁定持续跟同一人 |
| `center_largest_lock` | 同上，且强制开启锁定（等同 `--face_target_lock 1`） |
| `largest` | 始终面积最大（旧行为，两人差不多大时易左右跳） |
| `center` | 始终最靠近画面中心（不看面积） |

```bash
# real：居中说话人 + 锁定（默认，无需额外参数）
python main.py --face_detector mediapipe --face_target_policy center_largest \
  --save_face_overlay_video_dir ./测试结果_视频

# 关闭锁定，每帧按 center_largest 重选（旧行为，易抖）
python main.py --face_detector mediapipe --face_target_lock 0

# 恢复按最大脸选目标
python main.py --face_detector mediapipe --face_target_policy largest --face_target_lock 0
```

### 3.3 `main.py` 示例

```bash
# MediaPipe + 嘴中心裁切 + 保存人脸/叠加视频
# 建议用这个
python main.py --type torch_jit \
  --face_detector mediapipe \
  --mediapipe_lip_crop 1 \
  --mediapipe_lip_crop_scale 0.8 \
  --save_face_video_dir 测试结果_视频

# MediaPipe 人脸框裁剪（不用嘴关键点）
python main.py --type torch_jit --face_detector mediapipe --mediapipe_lip_crop 0

# OpenCV Haar
python main.py --type torch_jit --face_detector haar
```

| 参数 | 默认 | 说明 |
|------|------|------|
| `--face_detector` | `mediapipe` | `mediapipe` 或 `haar` |
| `--face_detector_model` | `detector.tflite` | MediaPipe 模型路径 |
| `--mediapipe_lip_crop` | `1` | 非 0 开启嘴中心裁剪 |
| `--mediapipe_lip_crop_scale` | `0.8` | 嘴裁边长相对 `last_box` 边长的比例 |
| `--mediapipe_lip_crop_min_px` | `48` | 嘴裁在原图上的最小边长（像素） |
| `--mediapipe_lip_crop_max_px` | `2048` | 嘴裁在原图上的最大边长（像素） |
| `--face_target_policy` | `center_largest` | 多人脸目标策略（见 §3.2.1） |
| `--face_target_lock` | `1` | 首帧选人后按 IoU 锁定同一人 |
| `--face_target_lock_min_iou` | `0.15` | 锁定跟随时最低 IoU，低于则保持上一帧框 |
| `--save_face_video_dir` | 空 | `{id}_face.mp4`（模型参考 crop） |
| `--save_face_overlay_video_dir` | `./测试结果_视频` | `{id}_face_overlay.mp4`（原图 + 绿/黄框） |

调嘴唇视野大小：优先 `--mediapipe_lip_crop_scale`（越大包含越多脸周区域）。

### 3.4 SDK 用法

```python
from stream_inference_SDK import StreamInferenceSDK

streamer = StreamInferenceSDK(
    ts_path="checkpoints/AV_Mossformer/torch_jit.zip",
    face_detector="mediapipe",
    face_detector_model_path="detector.tflite",
    mediapipe_use_lip_center_crop=1,
    mediapipe_lip_crop_scale=0.8,
)
```

未传参时 SDK 默认 `face_detector="haar"`、`mediapipe_use_lip_center_crop=0`；经 `main.py` 启动时以 `main.py` 的 `sdk_kwargs` 为准。

### 3.5 相关代码

| 文件 | 作用 |
|------|------|
| `face_mediapipe_tracker.py` | MediaPipe 检测、框/分数/嘴点、裁剪 |
| `av_stream_inference.py` | 选择 tracker、`face_crop_size`、`face_detect_every_n` 等 |
| `facedetection.py` | 静态图检测可视化 |
| `eval_face_metrics.py` | 评估：`--face-detector mediapipe`、`--mediapipe-lip-crop` |

## 4. ONNX：动态 vs 定长

仓库内默认 [`av_mossformer2.onnx`](checkpoints/AV_Mossformer/av_mossformer2.onnx) 导出时带 **动态轴**（`mixture` / `ref` 的时间维可变）。流式推理每次 hop 窗口长度不同，动态模型可直接匹配。

**定长 ONNX**（`av_mossformer2_fixed.onnx`）与 `main.py` 默认流式窗对齐：

- `infer_chunk_ms=500`、`context_ms=100`、`lookahead=0`、`audio_sr=16000`、`ref_sr=30`
- 固定输入：`mixture [1, 9600]`，`ref [1, 18, 96, 96, 3]`
- 推理时短于定长的输入会 **右侧补零**，长于定长会 **截断**（含 full-buffer 模式下的整段 ring buffer）

### 3.2.2 RTF 与稳定性（如何收回 ~0.1）

| 因素 | 对 RTF | 对 SI-SDR / 人脸稳定 |
|------|--------|----------------------|
| `--use_stream_cache 1`（默认） | 每 hop 对 ring **整段** forward，比 `0` 慢约 **0.05~0.15** | **必须保持 1** 才有 ~5 dB；`0` 约 ~2 dB |
| `--save_face_overlay_video_dir` 默认开启 | 主线程 `frame.copy` + 后台编码，常占 **0.05~0.15** | 与分离质量无关，测速应关 |
| Case B 用 `chunk_ms=100` 而 `infer_chunk_ms=500` | 上游调用次数约 **5×**，人脸裁剪重复执行 | `case_chunk_ms` 与 `infer_chunk_ms` 对齐（`main.py` 已默认） |
| `--face_target_lock 1` | 可忽略（仅检测帧多一次 IoU） | 多人脸不跳框 |
| MediaPipe + 嘴裁 | 人脸侧主要耗时 | 与模型质量相关 |

**测真实推理 RTF（推荐）**：

```bash
python main.py --bench_rtf --type torch_jit
# 或需要 overlay 时单独开，并加大 stride：
python main.py --save_face_overlay_video_dir ./测试结果_视频 --overlay_write_stride 10 --overlay_scale 0.5
```

启动日志应含 `[infer] backend=... use_stream_cache=1`；若 SI-SDR 要对齐 torch，**不要**为省 RTF 把 `use_stream_cache` 改成 `0`。

**`use_stream_cache`（与 torch 对齐，影响 SI-SDR 很大）**

| 值 | 行为 | 典型 SI-SDR |
|----|------|-------------|
| `1`（默认） | 对 ring buffer **整段** forward，再按 hop 切片 | ~5.x |
| `0` | 每个 hop 单独滑窗 forward | ~2.x |

此前 ONNX 因缺少 `trim_stream_cache` 会被**误降级**为 `0`；现已修复：在 `config.yaml` 的 `stream_cache_enable: 0`（无内部 KV cache）时，ONNX 与 torch 一样可使用 `--use_stream_cache 1`。启动时会打印实参，例如 `[infer] backend=onnx use_stream_cache=1 ...`。

导出定长模型（需已安装 `requirements-infer-only.txt`）：

```bash
python 脚本/export_onnx.py --fixed \
  --fp32_out checkpoints/AV_Mossformer/av_mossformer2_fixed.onnx \
  --context_ms 100 --infer_chunk_ms 500 --audio_sr 16000 --ref_sr 30 --image_size 96
```

若修改 `main.py` 的 `--infer_chunk_ms` / `--context_ms`，须用相同参数重新导出定长 ONNX。

推理示例：

```bash
# 推荐：与 torch 相同质量（默认 full-buffer）
python main.py --type onnx --use_stream_cache 1

# 定长 ONNX（须先按相同 context_ms/infer_chunk_ms 导出）
python main.py --type onnx --onnx_fixed --use_stream_cache 1

# 低延迟/调试滑窗（SI-SDR 会明显下降，仅在与 torch --use_stream_cache 0 对齐时使用）
python main.py --type onnx --use_stream_cache 0
```

INT8/FP16 需在定长 FP32 导出后重新量化（`export_onnx.py` 会按定长窗生成校准数据）。

## 5. TorchScript 定长（LibTorch C++）

[`export_jit.py`](export_jit.py) 的 trace 窗长已与 [`export_onnx.py`](export_onnx.py) 对齐：`T_audio = context + hop + lookahead`（默认 **9600** 采样点，**18** 帧 ref @30fps），与 [`av_mossformer2_fixed.onnx`](checkpoints/AV_Mossformer/av_mossformer2_fixed.onnx) 相同。

| 产物 | 说明 |
|------|------|
| `torch_jit.zip` | 默认 FP32 trace（修正后按 9600/18 trace） |
| `torch_jit_fixed.zip` | 显式定长副本，供 C++ 固定 shape 部署（`--export_fixed` 生成） |

导出（与 main 默认 `context_ms=100`、`infer_chunk_ms=500` 一致）：

```bash
python export_jit.py --context_ms 100 --infer_chunk_ms 500 --export_fixed \
  --script_mode trace --skip_quant --skip_fp16
```

Python 推理：

```bash
python main.py --type torch_jit --torch_jit_fixed --use_stream_cache 1
# 或
python main.py --type torch_jit --ts_path checkpoints/AV_Mossformer/torch_jit_fixed.zip --use_stream_cache 1
```

**C++ LibTorch**：始终向模型喂 `mixture [1,9600]`、`ref [1,18,96,96,3]`；ring buffer 不足则右侧补零、超出则截断。流式质量仍建议 **full-buffer**（`use_stream_cache=1`），与 ONNX 定长说明相同。修改 `infer_chunk_ms` / `context_ms` 后须用相同参数重导 ONNX 与 JIT。
