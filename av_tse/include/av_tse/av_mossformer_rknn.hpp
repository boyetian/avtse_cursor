#pragma once

#include <Eigen/Dense>
#include <opencv2/core.hpp>

#include <memory>
#include <string>
#include <vector>

#include "asr_frontend/rknn_session.hpp"

namespace av_tse {

/// RKNN separator: mixture [1,T] + ref_feat [1,C,Tv] (split deploy).
/// Legacy full-model RKNN (5D RGB ref) is no longer supported — use AvMossformerSplit.
class AvMossformerRknn {
 public:
  AvMossformerRknn(const std::string& rknn_path, int audio_len = 0, int ref_frames = 0,
                   int image_size = 96);

  void clearStreamCache() {}

  Eigen::VectorXf run(const Eigen::Ref<const Eigen::VectorXf>& mixture,
                      const std::vector<cv::Mat>& ref_frames);

  Eigen::VectorXf runWithRefFeat(const Eigen::Ref<const Eigen::VectorXf>& mixture,
                                 const std::vector<float>& ref_feat);

  int fixedAudioLen() const { return audio_len_; }
  int fixedRefFrames() const { return ref_frames_; }
  int refFeatChannels() const { return ref_feat_channels_; }

 private:
  std::unique_ptr<asr_frontend::RknnSession> session_;
  int audio_len_ = 0;
  int ref_frames_ = 0;
  int ref_feat_channels_ = 96;
  int image_size_ = 96;
};

}  // namespace av_tse
