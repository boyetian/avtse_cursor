#pragma once

#include <algorithm>
#include <cmath>
#include <opencv2/imgproc.hpp>

namespace av_tse {

template <typename Model>
void warmupIngressForward(Model& model, int image_size, int audio_sr, float ref_sr, int hop_samples,
                          int context_samples, int lookahead_samples) {
  constexpr float kMean = 0.506362f;
  constexpr float kStd = 0.272877f;
  const int n_vid = std::max(128, static_cast<int>(std::ceil(6.0 * ref_sr)));
  int audio_len = static_cast<int>(std::floor(static_cast<float>(n_vid) / ref_sr * audio_sr)) + 4096;
  audio_len = std::max(audio_len, context_samples + hop_samples + lookahead_samples + 1024);
  Eigen::VectorXf audio = Eigen::VectorXf::Random(audio_len) * 1e-2f;
  cv::Mat neutral(image_size, image_size, CV_32FC3, cv::Scalar(128.f, 128.f, 128.f));
  neutral = (neutral - kMean) / kStd;
  std::vector<cv::Mat> video_list(static_cast<size_t>(n_vid), neutral);
  if (audio.size() < hop_samples || static_cast<int>(video_list.size()) < 2) {
    return;
  }
  (void)model.run(audio, video_list);
}

}  // namespace av_tse
