#pragma once

#include <onnxruntime_cxx_api.h>

#include <Eigen/Dense>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace asr_frontend {

/// ZipEnhancer `zipenhancer_full.onnx` (onnx_full path), aligned with Python
/// `OnnxModel` + `ANSZipEnhancer::decode_stream_data`.
class ZipEnhancerFullOnnx {
 public:
  explicit ZipEnhancerFullOnnx(const std::string& onnx_path, float decode_window_sec = 1.0f,
                               int sample_rate = 16000, int intra_op_threads = 16);

  /// `bct` is row-major floats shaped `[B][C][T]` (batch, channel, time).
  std::vector<Eigen::VectorXf> decode_stream(const float* bct, int B, int C, int T,
                                              bool is_first, bool is_last);

 private:
  static Eigen::VectorXf audio_norm_row(const Eigen::Ref<const Eigen::VectorXf>& x);
  Eigen::MatrixXf preprocess(const float* bct, int B, int C, int T) const;

  Ort::Env env_;
  Ort::SessionOptions session_options_;
  std::unique_ptr<Ort::Session> session_;
  Ort::AllocatorWithDefaultOptions allocator_;

  std::vector<std::string> owned_input_names_;
  std::vector<std::string> owned_output_names_;
  std::vector<const char*> input_names_;
  std::vector<const char*> output_names_;

  float decode_window_sec_{};
  int sample_rate_{};
};

}  // namespace asr_frontend
