#pragma once

#include <Eigen/Dense>
#include <opencv2/core.hpp>

#include <vector>

#include "av_tse/av_mossformer_inference.hpp"

namespace av_tse {

int toVideoIndex(int sample_idx, int audio_sr, float ref_sr);

std::pair<Eigen::VectorXf, std::vector<cv::Mat>> alignAudioVideoList(
    const Eigen::Ref<const Eigen::VectorXf>& audio, const std::vector<cv::Mat>& video_list,
    int audio_sr, float video_fps);

std::pair<Eigen::VectorXf, std::vector<cv::Mat>> applyAvOffset(
    const Eigen::Ref<const Eigen::VectorXf>& audio, const std::vector<cv::Mat>& video_list,
    int audio_sr, float video_fps, float av_offset_ms);

struct HopRunResult {
  std::vector<Eigen::VectorXf> segments;
  int produced_samples = 0;
};

HopRunResult runNewHopsNonoverlap(const Eigen::Ref<const Eigen::VectorXf>& wav_al,
                                  const std::vector<cv::Mat>& vid_norm_list, AvMossformerModel& model,
                                  int hop_samples, int context_samples, int lookahead_samples,
                                  int audio_sr, float ref_sr, int produced_samples,
                                  bool use_stream_cache);

}  // namespace av_tse
