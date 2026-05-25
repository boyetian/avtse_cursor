#pragma once

#include <rknn_api.h>

#include <Eigen/Dense>
#include <memory>
#include <string>
#include <vector>

#include "asr_frontend/rknn_session.hpp"

namespace asr_frontend {

/// ZipEnhancer full RKNN backend, aligned with `ZipEnhancerFullOnnx`.
class ZipEnhancerFullRknn {
 public:
  explicit ZipEnhancerFullRknn(const std::string& rknn_path, float decode_window_sec = 1.0f,
                               int sample_rate = 16000);

  std::vector<Eigen::VectorXf> decode_stream(const float* bct, int B, int C, int T,
                                              bool is_first, bool is_last);

 private:
  static Eigen::VectorXf audio_norm_row(const Eigen::Ref<const Eigen::VectorXf>& x);
  Eigen::MatrixXf preprocess(const float* bct, int B, int C, int T) const;

  std::unique_ptr<RknnSession> session_;
  float decode_window_sec_{};
  int sample_rate_{};
  rknn_tensor_type input_type_{RKNN_TENSOR_FLOAT32};
  rknn_tensor_format input_fmt_{RKNN_TENSOR_UNDEFINED};
};

}  // namespace asr_frontend
