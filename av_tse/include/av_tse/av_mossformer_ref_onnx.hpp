#pragma once

#include <Eigen/Dense>
#include <onnxruntime_cxx_api.h>
#include <opencv2/core.hpp>

#include <memory>
#include <string>
#include <vector>

namespace av_tse {

/// Visual ref_encoder ONNX: gray [1,T,H,W] -> ref_feat [1,C,T] (CPU/ORT).
class AvMossformerRefOnnx {
 public:
  AvMossformerRefOnnx(const std::string& onnx_path, int num_threads = 4);

  std::vector<float> runGrayFrames(const std::vector<cv::Mat>& gray_frames, int image_size);

  int fixedRefFrames() const { return ref_frames_; }
  int refFeatChannels() const { return ref_feat_channels_; }

 private:
  Ort::Env env_;
  Ort::SessionOptions session_options_;
  Ort::AllocatorWithDefaultOptions allocator_;
  std::unique_ptr<Ort::Session> session_;
  std::vector<std::string> owned_input_names_;
  std::vector<std::string> owned_output_names_;
  std::vector<const char*> input_names_;
  std::vector<const char*> output_names_;
  int ref_frames_ = 18;
  int ref_feat_channels_ = 96;
  int image_size_ = 96;
};

}  // namespace av_tse
