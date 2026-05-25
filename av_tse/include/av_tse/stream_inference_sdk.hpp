#pragma once

#include <Eigen/Dense>
#include <opencv2/core.hpp>

#include <memory>
#include <string>
#include <vector>

#include "av_tse/face_haar_tracker.hpp"

namespace av_tse {

struct StreamInferenceSDKOptions {
  std::string config_yaml;
  std::string onnx_path;
  std::string ref_onnx_path;
  std::string rknn_path;
  int onnx_num_threads = 8;
  /// SDK FIFO block duration (e.g. 500 in main.py).
  float infer_chunk_ms = 200.f;
  /// Core model hop; 0 = use infer_chunk_ms. Prefer 200ms hop with 500ms SDK FIFO.
  float core_infer_chunk_ms = 0.f;
  float context_ms = 100.f;
  float max_history_ms = 100.f;
  int use_stream_cache = 1;
  float default_fps = 30.f;
  FaceHaarTrackerOptions face_tracker{};
};

class StreamInferenceSDK {
 public:
  explicit StreamInferenceSDK(const StreamInferenceSDKOptions& opt);
  ~StreamInferenceSDK();

  int audioSr() const;

  std::vector<Eigen::VectorXf> processAvStream(const Eigen::Ref<const Eigen::VectorXf>& audio_mono,
                                               const std::vector<cv::Mat>& video_bgr_uint8,
                                               bool is_start = false, bool is_end = false,
                                               int sampling_rate = 16000, float fps = 25.f);

  void close();

  /// Accept (C,T) audio; mixes to mono then calls processAvStream.
  static Eigen::VectorXf audioToMono(const Eigen::Ref<const Eigen::MatrixXf>& wav);

 private:
  static std::vector<cv::Mat> validateFrames(const std::vector<cv::Mat>& frames);

  StreamInferenceSDKOptions opt_;
  std::unique_ptr<class AVStreamInference> core_;
  float infer_chunk_ms_ = 200.f;
  float default_fps_ = 30.f;
  Eigen::VectorXf audio_buf_;
  std::vector<cv::Mat> video_buf_;
  bool need_core_start_ = true;
};

}  // namespace av_tse
