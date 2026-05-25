#pragma once

#include <Eigen/Dense>
#include <opencv2/core.hpp>

#include <memory>
#include <string>
#include <vector>

#include "av_tse/av_mossformer_inference.hpp"
#include "av_tse/av_tse_config.hpp"
#include "av_tse/face_haar_tracker.hpp"
#include "av_tse/incremental_video_resampler.hpp"

namespace av_tse {

struct AVStreamInferenceOptions {
  std::string config_yaml;
  std::string onnx_path;
  /// ORT ref_encoder (gray 4D); with rknn_path uses split deploy (ref CPU + sep NPU).
  std::string ref_onnx_path;
  std::string rknn_path;
  int onnx_num_threads = 8;
  float context_ms = 100.f;
  float infer_chunk_ms = 200.f;
  int lookahead_frames = 0;
  float max_history_ms = 100.f;
  int use_stream_cache = 1;
  int ingress_warmup = 1;
  float av_offset_ms = 0.f;
  int clear_cache_on_scene_switch = 1;
  float scene_switch_iou_thr = 0.15f;
  FaceHaarTrackerOptions face_tracker{};
};

class AVStreamInference {
 public:
  explicit AVStreamInference(const AVStreamInferenceOptions& opt);

  int audioSr() const { return audio_sr_; }

  void reset();
  void close();

  std::vector<Eigen::VectorXf> streamInference(const Eigen::Ref<const Eigen::VectorXf>& audio_chunk,
                                               const std::vector<cv::Mat>& video_chunk,
                                               bool is_start = false, bool is_end = false,
                                               int sampling_rate = 16000);

 private:
  Eigen::VectorXf normalizeAudioChunk(const Eigen::Ref<const Eigen::VectorXf>& audio_chunk,
                                        int sampling_rate);
  static std::vector<cv::Mat> normalizeVideoChunk(const std::vector<cv::Mat>& video_chunk);

  AVStreamInferenceOptions opt_;
  AvTseConfig cfg_;
  std::unique_ptr<AvMossformerModel> model_;

  static constexpr float kMean = 0.506362f;
  static constexpr float kStd = 0.272877f;

  int audio_sr_ = 16000;
  float ref_sr_ = 30.f;
  int image_size_ = 96;
  std::string backbone_;
  int hop_samples_ = 3200;
  int context_samples_ = 0;
  int lookahead_samples_ = 0;
  bool use_stream_cache_ = true;
  int max_history_samples_ = 0;
  int drop_unit_samples_ = 1;
  int drop_unit_ref_ = 1;
  int drop_unit_enc_ = 1;

  FaceHaarStreamTracker tracker_;
  IncrementalVideoResampler resampler_;
  Eigen::VectorXf audio_buf_;
  int produced_samples_ = 0;
  bool started_ = false;
};

}  // namespace av_tse
