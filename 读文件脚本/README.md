# 预处理 — 三层流水线第 1 层

将 8 通道 PCM + MP4 视频转换为标准格式 `a_i`, `v_i`，供下游人脸门控消费。

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

每个 PCM+MP4 对输出一个 npz 文件，对应一次 100ms chunk：

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
| `audio` | `(1, T)` | float32 | a_i：混合后的单通道音频 |
| `frames` | `(N, H, W, 3)` | uint8 | v_i：BGR 视频帧 |
| `sr` | `[16000]` | int | 采样率 |
| `fps` | `[24.0]` | float | 帧率 |

### 下游使用（流式 import）

```python
from prepare_chunk_input import read_8ch_pcm, mix_channels
import numpy as np

# 模拟流式：每 100ms 到达一个 chunk
raw_pcm = read_8ch_pcm("chunk_001.pcm", sr=16000, dtype="int16")  # (8, 1600) float32
a_i = mix_channels(raw_pcm, mode="clean")                          # (1600,) float32 mono

# a_i + v_i → 送入流水线第 2 层（人脸门控）
```

完整三层串联见项目根目录 [README.md](../README.md)。
