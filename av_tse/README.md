# C++ `av_tse`（ONNX + Eigen + OpenCV）

与 Python 版 [`av_tse`](../../av_tse/) 流式 SDK 对齐的**同步** C++ 实现，对外暴露与 `StreamInferenceSDK::process_av_stream` 等价的 C++ 接口。

| 能力 | 实现 |
|------|------|
| 模型推理 | ONNX Runtime 或 RKNN（`av_mossformer2.onnx` / `.rknn`） |
| 音频缓冲 / 矩阵 | Eigen |
| 人脸检测与裁剪 | OpenCV Haar + 流式跟踪 |
| 音视频对齐 / hop 切分 | 与 `av_stream_inference.py` 一致 |
| 采样率转换 | libsoxr |
| 配置 | yaml-cpp 读取 `config.yaml` |

推理后端在**编译期**选择：

| 后端 | CMake | 模型 |
|------|-------|------|
| ONNX（默认） | `-DAV_TSE_INFERENCE_BACKEND=ONNX` | `av_mossformer2.onnx` |
| RKNN（RK3588 等） | `-DAV_TSE_INFERENCE_BACKEND=RKNN` | `av_mossformer2.rknn` |

RK3588 Android 部署（模型转换、交叉编译、adb）见 **[`docs/RK3588_ANDROID_RKNN.md`](docs/RK3588_ANDROID_RKNN.md)**。

不含 PyTorch / TorchScript 路径。

---

## 架构概览

```text
业务侧 (C++)
    │
    ▼
StreamInferenceSDK::processAvStream()     ← 对外 API（FIFO 组包）
    │  infer_chunk_ms 切块（如 500ms）
    ▼
AVStreamInference::streamInference()      ← 核心流水线
    ├─ FaceHaarStreamTracker              ← BGR 帧 → 人脸 RGB crop
    ├─ IncrementalVideoResampler          ← 人脸序列 → ref_sr 时间轴
    ├─ alignAudioVideoList()              ← 音视频长度对齐
    └─ AvMossformerOnnx::run()            ← mixture + ref → 分离音频
```

一次 `processAvStream` 调用可能触发**零次、一次或多次**核心推理；返回 `std::vector<Eigen::VectorXf>`，每个元素为一段分离后的单声道 `float32` 波形，按时间顺序拼接即为输出。

---

## 目录结构与 Python 对应

```text
cpp/av_tse/
├── include/av_tse/
│   ├── stream_inference_sdk.hpp      # 对外 SDK（≈ stream_inference_SDK.py）
│   ├── av_stream_inference.hpp       # 核心推理（≈ av_stream_inference.py）
│   ├── av_mossformer_onnx.hpp        # ONNX 封装（≈ _ONNXModelWrapper）
│   ├── face_haar_tracker.hpp         # Haar 人脸跟踪
│   ├── incremental_video_resampler.hpp
│   ├── av_align.hpp                  # 对齐 / hop 切分
│   ├── audio_resampler.hpp           # libsoxr 重采样
│   └── av_tse_config.hpp             # config.yaml 解析
├── src/                              # 上述头文件实现
├── CMakeLists.txt
└── README.md
```

| C++ 类 / 文件 | Python |
|---------------|--------|
| `StreamInferenceSDK` | `StreamInferenceSDK` |
| `AVStreamInference` | `AVStreamInference` |
| `AvMossformerOnnx` | `_ONNXModelWrapper` |
| `FaceHaarStreamTracker` | `FaceHaarStreamTracker` |
| `IncrementalVideoResampler` | `IncrementalVideoResampler` |
| `alignAudioVideoList` / `runNewHopsNonoverlap` | `_align_audio_video_list` / `_run_new_hops_nonoverlap` |

静态库目标名：**`av_tse`**。由顶层 `cpp/CMakeLists.txt` 的 `add_subdirectory(av_tse)` 引入。

---

## 编译

在仓库 `cpp/` 目录下：

```bash
cd cpp
cmake -B build -S . -DASR_FRONTEND_BUILD_TESTS=ON
cmake --build build -j$(nproc)
```

产物：

| 目标 | 路径 |
|------|------|
| 静态库 | `build/av_tse/libav_tse.a` |
| 单元测试 | `build/bin/cpp_test_av_tse` |

### 依赖

| 依赖 | 说明 |
|------|------|
| **C++17** | 必须 |
| **Eigen3** | 未安装时 FetchContent 拉取 3.4.0 |
| **OpenCV** | `core` `imgproc` `objdetect` `videoio`（测试读视频还需 `imgcodecs`） |
| **ONNX Runtime** | 与 `asr_frontend` 共用；见下文 |
| **libsoxr** | 未安装时自动编译静态库 |
| **yaml-cpp** | FetchContent 0.8.0 |
| **GoogleTest** | 仅测试目标需要 |

### CMake 选项

| 选项 | 默认 | 说明 |
|------|------|------|
| `ASR_FRONTEND_BUILD_TESTS` | OFF | 设为 `ON` 以构建 `cpp_test_av_tse` |
| `AV_TSE_INFERENCE_BACKEND` | ONNX | `ONNX` 或 `RKNN`（RKNN 需 `RKNN_MODEL_ZOO_ROOT` 或 `RKNN_RKNPU2_ROOT`） |
| `AV_TSE_ANDROID_OPENCV_DIR` | 空 | **Android 专用**：自编译 OpenCV 的 abi 目录（`.../sdk/native/jni/abi-arm64-v8a`），见 [`scripts/build_opencv_android.sh`](scripts/build_opencv_android.sh) |
| `AV_TSE_FETCH_OPENCV` | ON | 无系统 OpenCV 时拉取并编译 OpenCV 4.8.0 最小集（首次较慢） |
| `ONNXRUNTIME_ROOT` | 自动 | 指向含 `include/`、`lib/` 的 ORT 根目录；可与 `asr_frontend` 共用 `build/onnxruntime-linux-x64-1.17.1` |

RK3588 Android RKNN 全流程见 [`docs/RK3588_ANDROID_RKNN.md`](docs/RK3588_ANDROID_RKNN.md)。板端 Case B（`av_tse_caseb`）视频仅使用预抽帧目录 `video/test03_frames/`，见该文档 §4.0。

**使用系统 OpenCV（推荐，编译更快）：**

```bash
sudo apt install libopencv-dev libsoxr-dev
cmake -B build -S . -DAV_TSE_FETCH_OPENCV=OFF -DASR_FRONTEND_BUILD_TESTS=ON
```

**指定已有 ONNX Runtime：**

```bash
export ONNXRUNTIME_ROOT=/path/to/onnxruntime-linux-x64-1.17.1
cmake -B build -S . -DASR_FRONTEND_BUILD_TESTS=ON
```

### 链接到你的工程

```cmake
add_subdirectory(cpp/av_tse)   # 或通过顶层 cpp 已 add_subdirectory
target_link_libraries(your_app PRIVATE av_tse)
target_include_directories(your_app PRIVATE cpp/av_tse/include)
```

运行时需要能找到 **ONNX Runtime** 动态库（若未静态链接）：

```bash
export LD_LIBRARY_PATH=/path/to/onnxruntime/lib:$LD_LIBRARY_PATH
```

---

## 模型与配置

路径相对于仓库内 **`av_tse/`** 目录（与 Python 工程一致）：

| 文件 | 说明 |
|------|------|
| `checkpoints/AV_Mossformer/config.yaml` | `audio_sr`、`ref_sr`、`image_size`、`backbone` 等 |
| `checkpoints/AV_Mossformer/av_mossformer2.onnx` | FP32 ONNX 模型 |

`config.yaml` 中常用字段（由 `AvTseConfig` 读取）：

- `audio_sr`：模型音频采样率（通常 **16000**）
- `ref_sr`：参考视频时间轴帧率（通常 **30**，与 `IncrementalVideoResampler` 的 target_fps 一致）
- `network_ref.image_size`：参考人脸输入边长（通常 **96**）

ONNX 输入名（与 Python 一致）：

- `mixture`：`[batch, T]` float32
- `ref`：`[batch, Tv, H, W, 3]` float32（时间维为参考帧数）

输出：分离后波形，形状约为 `[1, T]`。

---

## 对外 API：`StreamInferenceSDK`

头文件：`include/av_tse/stream_inference_sdk.hpp`

### `StreamInferenceSDKOptions`

| 字段 | 默认 | 说明 |
|------|------|------|
| `config_yaml` | — | `config.yaml` 路径（必填） |
| `onnx_path` | — | `.onnx` 路径（ONNX 后端必填） |
| `rknn_path` | — | `.rknn` 路径（RKNN 后端必填） |
| `onnx_num_threads` | 8 | ONNX Runtime 线程数 |
| `infer_chunk_ms` | 200 | **SDK FIFO** 出块时长（ms）；`main.py` 常用 **500** |
| `core_infer_chunk_ms` | 0 | **核心 hop** 时长；**0** 表示与 `infer_chunk_ms` 相同 |
| `context_ms` | 100 | 滑动窗口左上下文（ms） |
| `max_history_ms` | 100 | 环形缓冲最大历史（ms）；见下文建议 |
| `use_stream_cache` | 1 | 为 1 时语义与 Python 一致；**ONNX 无 `trim_stream_cache` 时会自动降为滑动窗口** |
| `default_fps` | 30 | `fps` 参数无效时的默认视频帧率 |
| `face_tracker` | 见 `FaceHaarTrackerOptions` | 人脸检测参数 |

### `processAvStream`

```cpp
std::vector<Eigen::VectorXf> processAvStream(
    const Eigen::Ref<const Eigen::VectorXf>& audio_mono,  // (T,) float32 单声道
    const std::vector<cv::Mat>& video_bgr_uint8,          // 每帧 H×W×3 BGR uint8
    bool is_start = false,
    bool is_end = false,
    int sampling_rate = 16000,
    float fps = 25.f);
```

**语义（与 Python `process_av_stream` 一致）：**

1. 上游可传入任意时长的 A/V chunk（如 100ms）；SDK 内部用 **FIFO** 缓存。
2. 当缓存中音频 ≥ `infer_chunk_ms` 且视频帧数 ≥ `round(fps * infer_chunk_ms/1000)` 时，弹出一块送入核心推理。
3. `is_start=true`：清空 SDK 与核心状态，下一块带 `is_start` 进核心。
4. `is_end=true`：将**不足一块的尾巴**再推理一次（`is_end` 传入核心），然后清空 SDK 缓冲。
5. 返回值：本次调用新产生的分离音频段列表；业务侧按顺序 `concat` 即可。

**视频帧要求：**

- `cv::Mat`，`CV_8UC3`，BGR，与 OpenCV `imread` / `VideoCapture` 一致。
- 帧数应与当前音频 chunk 时长大致匹配（测试里按 `fps * chunk_duration` 从视频索引取帧）。

**采样率：**

- 若 `sampling_rate != config.audio_sr`，核心内会用 libsoxr 重采样到模型采样率。

### 完整调用示例

```cpp
#include "av_tse/stream_inference_sdk.hpp"
#include <opencv2/imgcodecs.hpp>

av_tse::StreamInferenceSDKOptions opt;
opt.config_yaml = "av_tse/checkpoints/AV_Mossformer/config.yaml";
opt.onnx_path   = "av_tse/checkpoints/AV_Mossformer/av_mossformer2.onnx";
opt.infer_chunk_ms  = 500.f;
opt.context_ms      = 100.f;
opt.max_history_ms  = 600.f;   // 见「参数建议」
opt.use_stream_cache = 1;

av_tse::StreamInferenceSDK sdk(opt);
const int sr = sdk.audioSr();  // 通常 16000

std::vector<Eigen::VectorXf> all_out;

for (int i = 0; i < num_chunks; ++i) {
  Eigen::VectorXf audio = ...;           // 本段单声道 float32
  std::vector<cv::Mat> frames = ...;     // 与本段对齐的 BGR 帧

  auto segs = sdk.processAvStream(
      audio, frames,
      /*is_start=*/(i == 0),
      /*is_end=*/(i == num_chunks - 1),
      sr, /*fps=*/30.f);

  for (auto& s : segs) {
    all_out.push_back(std::move(s));
  }
}

sdk.close();

// 拼接 all_out 得到整段分离音频
```

---

## 核心 API：`AVStreamInference`（进阶）

若业务侧**自行**按固定块大小组好 A/V，可直接使用 `AVStreamInference::streamInference`，跳过 SDK FIFO。

头文件：`include/av_tse/av_stream_inference.hpp`

与 Python `AVStreamInference.stream_inference` 对齐：人脸跟踪 → 重采样 → 对齐 → hop 推理 → 环形缓冲 trim。

---

## 流式处理要点

### SDK 两层块大小

| 层级 | 参数 | 典型值 |
|------|------|--------|
| SDK FIFO | `infer_chunk_ms` | 500 ms（`main.py`） |
| 核心 hop | `infer_chunk_ms` / `core_infer_chunk_ms` | 500 ms 或 200 ms |

`main.py` 仅设置 `infer_chunk_ms=500`，SDK 与核心均为 500ms。  
测试与调试中也可使用 **SDK=500ms + core=200ms**（`core_infer_chunk_ms=200`），更易在 A/V 略不齐时凑满 hop。

### ONNX 与 `use_stream_cache`

Python 在检测到模型**没有** `trim_stream_cache` 时会打印警告并设 `use_stream_cache=False`，改为**滑动窗口**逐 hop 调 ONNX。

C++ 行为相同：构造 `AVStreamInference` 时若仍为 ONNX 且 `use_stream_cache=1`，会强制关闭 stream cache 并输出：

```text
[WARN] ONNX backend has no trim_stream_cache; fallback to use_stream_cache=0
```

### 环形缓冲（`max_history_ms`）

长流推理时，`audio_buf` 超过 `max_history_samples` 会丢弃头部音频，并同步 `trimHead` 参考视频帧，同时 `produced_samples -= n_drop_audio`。

C++ 额外保护（保证与 Python 行为兼容且输出完整）：

- **首次推理前**（`produced_samples == 0`）不 trim，避免未处理音频被裁掉。
- **`is_end` 时**不 trim，避免尾包丢失。
- **trim 参考帧**时通过 `capRefTrimFrames` 保留覆盖当前 `audio_buf` 所需的最少 `tgt` 帧数。

### 音视频对齐

`alignAudioVideoList` 将音频裁到视频可覆盖长度，并按音频长度保留对应视频帧。  
实现上对 `keep_video` 使用 `ceil`，且**不再**做第二次 `floor` 音频截断，以减少整除误差导致的样本丢失。

---

## 参数建议

与 [`av_tse/main.py`](../../av_tse/main.py) Case B 对齐的推荐配置（`cpp_test_av_tse` 使用）：

| 参数 | 建议值 | 说明 |
|------|--------|------|
| `infer_chunk_ms` | **500** | 与 `main.py --infer_chunk_ms` 一致 |
| `core_infer_chunk_ms` | **0** | 与 SDK 块相同；若需更密 hop 可设 **200** |
| `context_ms` | **100** | 与 `main.py --context_ms` 一致 |
| `max_history_ms` | **600** | `main.py` 默认为 **100**；C++ ONNX 滑动窗口下建议 **≥600**，使 ring 约 **9600** 样本（约 20 帧 @30fps），稳态 `wav_al ≥ hop(8000)`。仅用 100ms 时首块人脸帧不足可能导致长时间无输出或输出偏短 |
| `use_stream_cache` | **1** | ONNX 下会自动降级，可保持为 1 |
| `fps` | 实测视频帧率 | 测试视频 `test03.mp4` 为 **30fps**（勿写死 25） |

`max_history_samples` 计算公式（与 Python 相同）：

```text
max(hop_samples + lookahead_samples + 1024,
    round(audio_sr * max_history_ms / 1000))
```

---

## 测试

构建并运行（镜像 `main.py` Case B：`test03.wav` + `test03.mp4`）。推理后端与 `AV_TSE_INFERENCE_BACKEND` 一致：`cpp_test_av_tse` 在 ONNX 下使用 `av_mossformer2.onnx`，在 RKNN 下使用 `av_mossformer2.rknn`（可用环境变量 `AV_TSE_TEST_MODEL_PATH` 覆盖）。

### ONNX（x86 开发机，默认）

```bash
cd cpp
cmake -B build -S . -DASR_FRONTEND_BUILD_TESTS=ON -DAV_TSE_INFERENCE_BACKEND=ONNX
cmake --build build --target cpp_test_av_tse -j$(nproc)

export LD_LIBRARY_PATH="$(pwd)/build/onnxruntime-linux-x64-1.17.1/lib"
# FetchContent OpenCV 无 FFmpeg 时需外部 ffmpeg 解码 mp4：
export FFMPEG_PATH=/path/to/ffmpeg   # 可选

./build/bin/cpp_test_av_tse --gtest_filter='AvTseMain.*'
```

### RKNN（aarch64 Linux 或板端）

x86 无法链接/运行 aarch64 的 `librknnrt.so`；请在 **aarch64** 主机上配置 RKNN 后构建同一测试目标：

```bash
cd cpp
cmake -B build-rknn -S . \
  -DASR_FRONTEND_BUILD_TESTS=ON \
  -DAV_TSE_INFERENCE_BACKEND=RKNN \
  -DRKNN_MODEL_ZOO_ROOT=/path/to/rknn_model_zoo
cmake --build build-rknn --target cpp_test_av_tse -j$(nproc)

export LD_LIBRARY_PATH=/path/to/rknn_model_zoo/3rdparty/rknpu2/Linux/aarch64
# 或 rknn-toolkit2: .../rknpu2/runtime/Linux/librknn_api/aarch64
./build-rknn/bin/cpp_test_av_tse --gtest_filter='AvTseMain.*'
```

需本地已有 `checkpoints/AV_Mossformer/av_mossformer2.rknn`（由 [`scripts/convert_av_mossformer_rknn.py`](scripts/convert_av_mossformer_rknn.py) 从 ONNX 转换）。

测试逻辑概要：

- 上游按 **100ms** 切分音频/视频（与 `run_case_b_full_numpy(..., chunk_ms=100)` 一致）。
- SDK `infer_chunk_ms=500`，核心 hop 500ms，`max_history_ms=600`。
- 统计 RTF，写出 `build/av_tse_test/test03_out.wav`。
- 断言输出时长与输入相差 **< 100ms**。

缺少 `av_tse/测试用例/` 或当前后端对应模型时，测试 **`GTEST_SKIP`**。

### 调试

```bash
export AV_TSE_DEBUG=1
./build/bin/cpp_test_av_tse --gtest_filter='AvTseMain.*'
```

会打印每步 `tgt` 帧数、`audio_buf` / `wav_al` 长度、`produced`、`hop`、`segs` 等，便于排查对齐与 ring trim 问题。

---

## 与 Python 的差异与限制

| 项目 | 说明 |
|------|------|
| 推理后端 | ONNX（默认）或 RKNN（`AV_TSE_INFERENCE_BACKEND=RKNN`）；无 torch / torch_jit |
| 异步 | 无；全同步 API |
| MP4 读取 | 测试依赖 OpenCV `VideoCapture`；无 FFmpeg 时需 `FFMPEG_PATH` 走命令行抽帧 |
| Haar 模型 | FetchContent OpenCV 时通过 `AV_TSE_HAAR_CASCADE_PATH` 编译进路径；系统 OpenCV 用自带 cascades |
| `max_history_ms` | 默认与 Python 相同为 100；**完整输出**建议 600（见上表） |
| `alignAudioVideoList` | C++ 去掉二次 audio floor；Python 仍保留，极端情况下 C++ 输出略长 |

---

## 常见问题

### 运行时报 `error while loading shared libraries: libonnxruntime.so`

```bash
export LD_LIBRARY_PATH=/path/to/onnxruntime/lib:$LD_LIBRARY_PATH
```

或使用 CMake 已为测试设置的 `BUILD_RPATH`（`build/onnxruntime-linux-x64-1.17.1/lib`）。

### OpenCV 无法打开 mp4

FetchContent 构建默认 **无 FFmpeg**。测试会回退到 `FFMPEG_PATH` 指向的 `ffmpeg` 抽帧。安装带 FFmpeg 的 OpenCV 或设置：

```bash
export FFMPEG_PATH=/home/user/FFmpeg/build/ffmpeg
```

### 输出时长明显短于输入

1. 检查 `max_history_ms` 是否过小（建议 600）。
2. 检查 `fps` 是否与视频一致。
3. 开启 `AV_TSE_DEBUG=1`，确认 `wav_al >= hop_samples` 是否经常成立。
4. 确认每段视频 chunk 含足够帧且人脸检测正常（`tgt` 帧数过少会缩短 `wav_al`）。

### 推理极慢 / RTF 高

ONNX 滑动窗口模式下每个 hop 单独 forward，RTF 通常 **> 1**（CPU）。可尝试：

- 增大 `onnx_num_threads`
- 使用 `infer_chunk_ms` 与 `core_infer_chunk_ms` 折中（更大 hop 减少 forward 次数）

### 构造失败 `failed to load OpenCV haarcascade`

安装 `libopencv-dev` 或确保 FetchContent OpenCV 的 `data/haarcascades/haarcascade_frontalface_default.xml` 存在。

---

## 参考

- Python 入口：[`av_tse/main.py`](../../av_tse/main.py)
- Python SDK：[`av_tse/stream_inference_SDK.py`](../../av_tse/stream_inference_SDK.py)
- C++ 测试：[`cpp/test/test_av_tse_main.cpp`](../test/test_av_tse_main.cpp)
- 同级模块：[`cpp/asr_frontend/README.md`](../asr_frontend/README.md)
