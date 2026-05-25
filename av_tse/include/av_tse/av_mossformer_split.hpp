#pragma once

#include "av_tse/av_mossformer_ref_onnx.hpp"
#include "av_tse/av_mossformer_rknn.hpp"

#include <Eigen/Dense>
#include <opencv2/core.hpp>

#include <memory>
#include <string>
#include <vector>

namespace av_tse {

/// Split deploy: ORT ref_encoder (gray) + RKNN separator (mixture + ref_feat).
class AvMossformerSplit {
 public:
  AvMossformerSplit(const std::string& ref_onnx_path, const std::string& rknn_sep_path,
                    int audio_len = 0, int ref_frames = 0, int image_size = 96,
                    int onnx_num_threads = 4);

  void clearStreamCache() {}

  Eigen::VectorXf run(const Eigen::Ref<const Eigen::VectorXf>& mixture,
                      const std::vector<cv::Mat>& ref_frames);

 private:
  static cv::Mat rgbToGray(const cv::Mat& rgb);
  std::unique_ptr<AvMossformerRefOnnx> ref_encoder_;
  std::unique_ptr<AvMossformerRknn> sep_rknn_;
  int image_size_ = 96;
};

}  // namespace av_tse
