#pragma once

#include <Eigen/Dense>
#include <opencv2/core.hpp>

#include <memory>
#include <string>
#include <vector>

#include "asr_frontend/rknn_session.hpp"

namespace av_tse {

/// AV-Mossformer RKNN wrapper with fixed mixture/ref shapes (pad or trim at runtime).
class AvMossformerRknn {
 public:
  AvMossformerRknn(const std::string& rknn_path, int audio_len = 0, int ref_frames = 0,
                   int image_size = 96);

  void clearStreamCache() {}

  Eigen::VectorXf run(const Eigen::Ref<const Eigen::VectorXf>& mixture,
                      const std::vector<cv::Mat>& ref_frames);

  int fixedAudioLen() const { return audio_len_; }
  int fixedRefFrames() const { return ref_frames_; }
  int imageSize() const { return image_size_; }

 private:
  std::unique_ptr<asr_frontend::RknnSession> session_;
  int audio_len_ = 0;
  int ref_frames_ = 0;
  int image_size_ = 96;
  int channels_ = 3;
};

}  // namespace av_tse
