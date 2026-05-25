#pragma once

namespace asr_frontend {

/// Shared DFSMN AEC streaming constants (ONNX and RKNN backends).
struct DfsmnAecConfig {
  static constexpr int kChunkSize = 16320;
  static constexpr int kStride = 16000;
  static constexpr int kSampleRate = 16000;
};

}  // namespace asr_frontend
