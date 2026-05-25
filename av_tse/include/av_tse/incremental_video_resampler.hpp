#pragma once

#include <opencv2/core.hpp>
#include <vector>

namespace av_tse {

class IncrementalVideoResampler {
 public:
  IncrementalVideoResampler(float src_fps, float target_fps, int image_size, float mean, float std,
                            const std::string& normalize_mode = "mossformer");

  const std::vector<cv::Mat>& tgtFrames() const { return tgt_frames_; }

  void appendSrcFacesRgb255(const std::vector<cv::Mat>& new_src_frames);

  /// Drops n target frames from head; returns number actually dropped.
  int trimHead(int n_target_frames);

 private:
  float src_fps_;
  float target_fps_;
  int image_size_;
  float mean_;
  float std_;
  std::string normalize_mode_;
  std::vector<cv::Mat> src_frames_;
  std::vector<cv::Mat> tgt_frames_;
  int src_dropped_ = 0;
  int tgt_dropped_ = 0;
};

}  // namespace av_tse
