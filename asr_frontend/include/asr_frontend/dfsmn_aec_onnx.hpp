#pragma once

#include <onnxruntime_cxx_api.h>

#include <Eigen/Dense>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace asr_frontend {

/// DFSMN AEC ONNX (`DFSMN_AEC_opt.onnx`), aligned with Python `OnnxModel` +
/// `CLS_DFMSN_AEC::decode_stream_data`.
class DfsmnAecOnnx {
 public:
  static constexpr int kChunkSize = 16320;
  static constexpr int kStride = 16000;
  static constexpr int kSampleRate = 16000;

  explicit DfsmnAecOnnx(const std::string& onnx_path, int intra_op_threads = 8);

  /// `nearend` / `farend` shape (C, T) float in roughly [-1, 1]. Mono int16
  /// output length matches Python `output[:total_len]` after mean/pad/trim.
  Eigen::Matrix<std::int16_t, Eigen::Dynamic, 1> decode_stream(
      const Eigen::Ref<const Eigen::MatrixXf>& nearend,
      const Eigen::Ref<const Eigen::MatrixXf>& farend);

 private:
  Ort::Env env_;
  Ort::SessionOptions session_options_;
  std::unique_ptr<Ort::Session> session_;
  Ort::AllocatorWithDefaultOptions allocator_;

  std::vector<std::string> owned_input_names_;
  std::vector<std::string> owned_output_names_;
  std::vector<const char*> input_names_;
  std::vector<const char*> output_names_;
};

}  // namespace asr_frontend
