# C++ `asr_frontend`（ONNX / RKNN + Eigen）

与 Python 版 `asr_frontend` 流式流水线对齐的同步 C++ 实现。默认在 Linux 上使用 **ONNX Runtime**；在 RK3588 Android 上可编译 **RKNN** 后端。缓冲与矩阵运算主要用 **Eigen**。

## 编译

在仓库根目录执行：

```bash
cd cpp
cmake -B build -S .
cmake --build build -j$(nproc)
```

依赖说明：

- **Eigen**：若系统未安装 `Eigen3`（无 `Eigen3Config.cmake`），CMake 会通过 `FetchContent` 自动拉取 Eigen 3.4.0。
- **ONNX Runtime**：在 Linux x64 上，若未设置 `ONNXRUNTIME_ROOT`，CMake 可将 CPU 版预编译包下载到 `build/onnxruntime-linux-x64-1.17.1/`（可用 `-DONNXRUNTIME_ROOT=/你的/onnxruntime路径` 或环境变量 `ONNXRUNTIME_ROOT` 覆盖）。关闭自动下载：`-DASR_FRONTEND_DOWNLOAD_ONNXRUNTIME=OFF`。
- **单元测试**：安装 `libgtest-dev`（或自行编译 GTest）后重新 `cmake`，会生成目标 `asr_frontend_tests`（仅 ONNX 后端）。

## RK3588 Android（RKNN）快速入门

1. 安装 `rknn-toolkit2`，转换模型：

```bash
cd cpp/asr_frontend/scripts
python3 convert_dfsmn_aec_rknn.py --target rk3588
python3 convert_zipenhancer_rknn.py --target rk3588
```

2. 交叉编译并安装到 `install/`：

```bash
export ANDROID_NDK_PATH=~/other/android-ndk-r19c
export RKNN_MODEL_ZOO_ROOT=~/workspace/rknn_model_zoo
./scripts/build_android_rk3588.sh
```

3. 推送到板子并运行：

```bash
adb push install/rk3588_android_arm64-v8a/asr_frontend_smoke /data/
adb shell 'cd /data/asr_frontend_smoke && export LD_LIBRARY_PATH=./lib && ./asr_frontend_smoke model/DFSMN_AEC_opt.rknn model/zipenhancer_full.rknn'
```

完整说明（环境、转换、adb 调试、故障排查）见 **[docs/RK3588_ANDROID_RKNN.md](docs/RK3588_ANDROID_RKNN.md)**。

**Android ONNX（CPU 调试，无需 RKNN 转换）：**

```bash
export ANDROID_NDK_PATH=~/other/android-ndk-r19c
export INFERENCE_BACKEND=ONNX
./scripts/build_android_rk3588.sh
adb push install/rk3588_android_arm64-v8a/asr_frontend_smoke /data/
adb shell 'cd /data/asr_frontend_smoke && export LD_LIBRARY_PATH=./lib && ./asr_frontend_smoke model/DFSMN_AEC_opt.onnx model/zipenhancer_full.onnx'
```

板端 GTest 见 **[../test/README.md](../test/README.md)**。

编译期通过 `-DASR_FRONTEND_INFERENCE_BACKEND=ONNX|RKNN` 选择后端（`build_android_rk3588.sh` 默认 `RKNN`，可用 `INFERENCE_BACKEND=ONNX` 切换）。

## 目录与 Python 对应关系

| C++ | Python 参考 |
|-----|----------------|
| `include/asr_frontend/stream_inference.hpp` | `asr_frontend/streamer/stream_inference.py`（核心逻辑对应 `_process_chunk_sync`） |
| `include/asr_frontend/stream_buffer_2d.hpp` | `asr_frontend/streamer/stream_buffer_2D.py` |
| `include/asr_frontend/asr_frontend_api.hpp` | `asr_frontend/asr_frontend_api.py`（此处为 ONNX 固定接线） |
| `include/asr_frontend/dfsmn_aec_onnx.hpp` | `dfsmn_aec/cls_dfsmn_aec.py` + `dfsmn_aec/models/onnx_model.py` |
| `include/asr_frontend/zipenhancer_full_onnx.hpp` | `zipenhancer/cls_zipenhancer_model.py`（`onnx_full` / `OnnxModel`） |
| `include/asr_frontend/dfsmn_aec_rknn.hpp` | 同上（RKNN 后端） |
| `include/asr_frontend/zipenhancer_full_rknn.hpp` | 同上（RKNN 后端） |

## 模型路径

默认与 Python `config.py` 一致（**相对当前进程工作目录**）：

- ONNX：`checkpoints/dfsmn_aec_16k/DFSMN_AEC_opt.onnx`、`checkpoints/zip_enhancer_se_16k/zipenhancer_full.onnx`
- RKNN：同目录下 `DFSMN_AEC_opt.rknn`、`zipenhancer_full.rknn`（需先转换）

可在 `StreamInferenceOptions::paths` 中覆盖对应字段（`dfsmn_aec_onnx` / `zipenhancer_full_onnx` 或 `dfsmn_aec_rknn` / `zipenhancer_full_rknn`）。

## 调用示例

```cpp
asr_frontend::StreamInferenceOptions opt;
asr_frontend::StreamInference si(opt);
Eigen::MatrixXf ne = ...; // 形状 (通道数, 采样点数)
auto outs = si.stream_inference(ne, std::nullopt, true, false, false, 16000);
```

## 与 Python 对齐时的注意点

1. **SE 后端**：C++ 固定加载 `zipenhancer_full.onnx`。Python 侧 `ASRFrontendAPI` 默认可能走 JIT；若要对齐数值，请在 Python 中同样走 ONNX（`onnx_full`）并使用同一路径权重。
2. **重采样**：当 `sampling_rate != target_sr` 时，C++ 使用**线性插值**重采样；Python 使用 `torchaudio.transforms.Resample`，两者在变速率场景下**不会**逐采样完全一致。
3. **数值容差**：对齐长度后，SE 输出可用 `np.allclose(..., atol=1e-3, rtol=1e-2)` 等做对比；AEC 在 int16 路径、输入一致时通常更接近。

## `ai.onnx.ml` opset 5 与模型加载失败

若加载 ONNX 时出现类似 **「Current official support for domain ai.onnx.ml is till opset 4」** 的报错，说明 ONNX Runtime 默认启用了「仅允许官方已发布 opset」的校验；而部分导出的 DFSMN 模型在 **`ai.onnx.ml` 域使用 opset 5**，会触发加载阶段直接失败。

本工程在构造 `AsrFrontendApi` 时会调用 `relax_onnx_released_opset_check_if_unset()`：**仅当环境变量尚未设置时**，将 `ALLOW_RELEASED_ONNX_OPSET_ONLY` 设为 `0`，从而允许加载（可能伴随警告，而不再抛异常）。

也可在运行任意可执行文件前自行设置：

```bash
export ALLOW_RELEASED_ONNX_OPSET_ONLY=0
```

若希望恢复严格校验，在进程启动前设置：

```bash
export ALLOW_RELEASED_ONNX_OPSET_ONLY=1
```

## 建议的对齐验证流程

在 Python 中导出固定音频块（如 `np.savez`），在小型 C++ 程序中读取同一份数据推理；或在 Python 脚本中加载相同 `.npz`，在「仅 ONNX」路径下对比输出形状与误差。
