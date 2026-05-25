#pragma once

#include <array>
#include <optional>
#include <utility>

#include <opencv2/core.hpp>
#include <opencv2/objdetect.hpp>

namespace av_tse {

struct FaceHaarTrackerOptions {
  int crop_size = 128;
  float face_scale = 1.25f;
  int detect_every_n = 5;
  int detect_max_side = 320;
  float haar_scale_factor = 1.15f;
  int haar_min_neighbors = 4;
  float box_smooth_alpha = 0.85f;
};

class FaceHaarStreamTracker {
 public:
  explicit FaceHaarStreamTracker(const FaceHaarTrackerOptions& opt = FaceHaarTrackerOptions{});

  /// Returns RGB float32 crop (0..255 scale) and whether scene switched.
  std::pair<cv::Mat, bool> processBgr(const cv::Mat& frame_bgr, float scene_switch_iou_thr = 0.15f);

 private:
  static float boxIouXyxy(const float* a, const float* b);

  FaceHaarTrackerOptions opt_;
  cv::CascadeClassifier detector_;
  std::optional<std::array<float, 4>> last_box_;
  std::optional<std::array<float, 4>> last_detected_box_;
  int frame_idx_ = 0;
};

}  // namespace av_tse
