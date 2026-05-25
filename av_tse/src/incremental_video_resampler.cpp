#include "av_tse/incremental_video_resampler.hpp"

#include <algorithm>
#include <cmath>

#include <opencv2/imgproc.hpp>

namespace av_tse {

IncrementalVideoResampler::IncrementalVideoResampler(float src_fps, float target_fps, int image_size,
                                                   float mean, float std,
                                                   const std::string& normalize_mode)
    : src_fps_(src_fps),
      target_fps_(target_fps),
      image_size_(image_size),
      mean_(mean),
      std_(std),
      normalize_mode_(normalize_mode) {}

void IncrementalVideoResampler::appendSrcFacesRgb255(
    const std::vector<cv::Mat>& new_src_frames) {
  if (new_src_frames.empty()) {
    return;
  }
  for (const auto& frm : new_src_frames) {
    src_frames_.push_back(frm);
  }

  const int src_len_abs = src_dropped_ + static_cast<int>(src_frames_.size());
  const float duration_sec =
      std::max(static_cast<float>(src_len_abs) / std::max(src_fps_, 1e-6f), 1.f / std::max(src_fps_, 1.f));
  const int tgt_len_total_abs =
      std::max(1, static_cast<int>(std::lround(duration_sec * target_fps_)));
  const int tgt_len_prev_abs = tgt_dropped_ + static_cast<int>(tgt_frames_.size());
  if (tgt_len_total_abs <= tgt_len_prev_abs) {
    return;
  }

  for (int ti_abs = tgt_len_prev_abs; ti_abs < tgt_len_total_abs; ++ti_abs) {
    int si_abs = static_cast<int>(std::lround((static_cast<float>(ti_abs) / target_fps_) * src_fps_));
    si_abs = std::clamp(si_abs, 0, src_len_abs - 1);
    int si_local = si_abs - src_dropped_;
    si_local = std::clamp(si_local, 0, static_cast<int>(src_frames_.size()) - 1);

    cv::Mat frm = src_frames_[static_cast<size_t>(si_local)];
    if (frm.rows != image_size_ || frm.cols != image_size_) {
      cv::resize(frm, frm, cv::Size(image_size_, image_size_), 0, 0, cv::INTER_AREA);
    }
    cv::Mat normed;
    frm.convertTo(normed, CV_32F, 1.0 / 255.0);
    if (normalize_mode_ == "mossformer") {
      normed = (normed - mean_) / std_;
    }
    tgt_frames_.push_back(normed);
  }
}

int IncrementalVideoResampler::trimHead(int n_target_frames) {
  int n_t = std::max(0, n_target_frames);
  n_t = std::min(n_t, static_cast<int>(tgt_frames_.size()));
  if (n_t <= 0) {
    return 0;
  }
  const int new_tgt_dropped_abs = tgt_dropped_ + n_t;
  int target_min_si_abs =
      static_cast<int>(std::floor(static_cast<float>(new_tgt_dropped_abs) /
                                  std::max(target_fps_, 1e-6f) * src_fps_)) -
      2;
  target_min_si_abs = std::max(0, target_min_si_abs);
  int n_s = std::max(0, target_min_si_abs - src_dropped_);
  n_s = std::min(n_s, static_cast<int>(src_frames_.size()));

  tgt_frames_.erase(tgt_frames_.begin(), tgt_frames_.begin() + n_t);
  tgt_dropped_ += n_t;
  if (n_s > 0) {
    src_frames_.erase(src_frames_.begin(), src_frames_.begin() + n_s);
    src_dropped_ += n_s;
  }
  return n_t;
}

}  // namespace av_tse
