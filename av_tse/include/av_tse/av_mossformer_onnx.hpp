#pragma once

#include <Eigen/Dense>
#include <onnxruntime_cxx_api.h>
#include <opencv2/core.hpp>

#include <memory>
#include <string>
#include <vector>

namespace av_tse {

class AvMossformerOnnx {
 public:
  explicit AvMossformerOnnx(const std::string& onnx_path, int num_threads = 8);

  void clearStreamCache() {}

  /// mixture [1,T], ref [1,Tv,H,W,3] row-major flat layout internally.
  Eigen::VectorXf run(const Eigen::Ref<const Eigen::VectorXf>& mixture,
                      const std::vector<cv::Mat>& ref_frames);

 private:
  Ort::Env env_;
  Ort::SessionOptions session_options_;
  Ort::AllocatorWithDefaultOptions allocator_;
  std::unique_ptr<Ort::Session> session_;
  std::vector<std::string> owned_input_names_;
  std::vector<std::string> owned_output_names_;
  std::vector<const char*> input_names_;
  std::vector<const char*> output_names_;
};

}  // namespace av_tse
