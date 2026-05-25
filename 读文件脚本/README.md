# scripts — 音视频预处理工具

## prepare_chunk_input.py

将 8 通道 PCM + MP4 视频转换为模型推理所需的格式。

### 8 通道 PCM 通道分配（1-indexed）

| 通道 | 内容 | 参与混合 |
|---|---|---|
| 1, 2 | 空 | 否 |
| 3, 4, 5, 6 | 有效 | 是（权重 1.0） |
| 7, 8 | 带噪声 | 否（clean）/ 是（noisy，权重 0.5） |

### 混合模式

- `clean`：只平均通道 3,4,5,6
- `noisy`：加权平均通道 3-8，3-6 权重 1.0，7-8 权重 0.5

### 用法

```bash
# 单个 chunk
python scripts/prepare_chunk_input.py \
  --pcm chunk_001.pcm --mp4 chunk_001.mp4 \
  --mode clean

# 批量处理（PCM 和 MP4 分开目录，按文件名匹配）
python scripts/prepare_chunk_input.py \
  --pcm_dir ./pcm/ --mp4_dir ./mp4/ \
  --mode clean --out_dir ./processed/
```

### 参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--pcm` | - | 单个 8 通道 PCM 文件路径 |
| `--mp4` | - | 单个 MP4 视频路径 |
| `--pcm_dir` | - | PCM 文件目录（批量模式） |
| `--mp4_dir` | - | MP4 文件目录（批量模式） |
| `--sr` | 16000 | 采样率 |
| `--dtype` | int16 | PCM 数据类型（可选 int32/float32） |
| `--mode` | clean | 通道混合模式（clean/noisy） |
| `--out_dir` | 与 PCM 同目录 | 输出目录 |

### 输出

每个 PCM+MP4 对输出一个 npz 文件，例如：

```
./pcm/                          ./processed/
  20260429102502.pcm              20260429102502_clean.npz
  20260429102503.pcm    →→→       20260429102503_clean.npz
./mp4/
  20260429102502.mp4
  20260429102503.mp4
```

npz 内容：

| key | shape | dtype | 说明 |
|---|---|---|---|
| `audio` | `(1, T)` | float32 | 混好的单通道音频 |
| `frames` | `(N, H, W, 3)` | uint8 | BGR 视频帧 |
| `sr` | `[16000]` | int | 采样率 |
| `fps` | `[25.0]` | float | 帧率 |

### 下游使用

```python
from scripts.prepare_chunk_input import load_chunk
import glob

# 加载各 chunk
chunks = []
for npz_file in sorted(glob.glob("./processed/*_clean.npz")):
    audio, frames, sr, fps = load_chunk(npz_file)
    chunks.append((audio, frames))

# Case A: 逐 chunk 喂入 SDK
from stream_inference_SDK import StreamInferenceSDK
streamer = StreamInferenceSDK()
outputs_all = run_case_a_stream_chunks(streamer, chunks, sr=sr, fps=fps)
streamer.close()
```
