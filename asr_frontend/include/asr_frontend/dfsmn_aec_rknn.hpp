#pragma once

#include <rknn_api.h>

#include <Eigen/Dense>
#include <cstdint>
#include <memory>
#include <string>

#include "asr_frontend/rknn_session.hpp"

namespace asr_frontend {

/// DFSMN AEC RKNN backend, aligned with `DfsmnAecOnnx`.
class DfsmnAecRknn {
 public:
  static constexpr int kChunkSize = 16320;
  static constexpr int kStride = 16000;
  static constexpr int kSampleRate = 16000;

  explicit DfsmnAecRknn(const std::string& rknn_path);

  Eigen::Matrix<std::int16_t, Eigen::Dynamic, 1> decode_stream(
      const Eigen::Ref<const Eigen::MatrixXf>& nearend,
      const Eigen::Ref<const Eigen::MatrixXf>& farend);

 private:
  std::unique_ptr<RknnSession> session_;
  rknn_tensor_type input_type_{RKNN_TENSOR_INT16};
  rknn_tensor_format input_fmt_{RKNN_TENSOR_UNDEFINED};
};

}  // namespace asr_frontend
