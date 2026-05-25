#pragma once

#include <Eigen/Dense>
#include <memory>
#include <optional>
#include <vector>

#include "asr_frontend/asr_frontend_api.hpp"
#include "asr_frontend/dfsmn_aec_config.hpp"
#include "asr_frontend/stream_buffer_2d.hpp"

namespace asr_frontend {

struct StreamInferenceOptions {
  bool enable_speech_enhancement = true;
  bool enable_aec = true;
  std::optional<int> nearend_channels;
  std::optional<int> farend_channels;
  std::optional<int> se_channels;
  bool enable_batch = false;
  AsrFrontendPaths paths{};
  /// ZipEnhancer streaming window in seconds (must match training config).
  float se_decode_window_sec = 1.0f;
};

/// Synchronous streaming pipeline matching Python `StreamInference` core
/// (`_process_chunk_sync`), without asyncio / worker thread.
class StreamInference {
 public:
  explicit StreamInference(const StreamInferenceOptions& opt = StreamInferenceOptions{});

  void reset();

  /// Each returned vector is one model output chunk (mono float in [-1, 1] for SE).
  std::vector<Eigen::VectorXf> stream_inference(
      const std::optional<Eigen::MatrixXf>& nearend,
      const std::optional<Eigen::MatrixXf>& farend, bool is_start = false, bool is_end = false,
      bool flush_buffer = false, int sampling_rate = 16000);

 private:
  void reset_state_();
  void init_buffers_(int nearend_ch, int farend_ch);

  Eigen::MatrixXf resample_if_needed_(const Eigen::Ref<const Eigen::MatrixXf>& audio,
                                      int sampling_rate);

  void push_and_process_(const Eigen::Ref<const Eigen::MatrixXf>& nearend,
                         const std::optional<Eigen::MatrixXf>& farend,
                         std::vector<Eigen::VectorXf>& outputs);

  Eigen::VectorXf run_aec_(const Eigen::Ref<const Eigen::MatrixXf>& nearend_win,
                            const Eigen::MatrixXf* farend_win, bool is_last);

  std::vector<Eigen::VectorXf> run_se_(const Eigen::Ref<const Eigen::VectorXf>& aec_mono,
                                       bool should_flush);

  static std::vector<Eigen::VectorXf> split_batch_output_(
      const std::vector<Eigen::VectorXf>& se_out);

  std::optional<Eigen::VectorXf> flush_aec_();

  StreamInferenceOptions opt_;
  std::unique_ptr<AsrFrontendApi> api_;

  static constexpr float kAecDecodeWindowSec =
      static_cast<float>(DfsmnAecConfig::kChunkSize) / static_cast<float>(DfsmnAecConfig::kSampleRate);
  static constexpr int kAecRingStride = DfsmnAecConfig::kStride;

  int target_sr_{16000};

  std::unique_ptr<StreamingRingBuffer2D> nearend_buffer_;
  std::unique_ptr<StreamingRingBuffer2D> farend_buffer_;
  std::unique_ptr<StreamingRingBuffer2D> se_buffer_;

  bool is_first_aec_{true};
  bool is_first_se_{true};
  std::optional<int> current_sr_;
  int se_ch_{1};
  std::optional<int> channels_;
  std::optional<int> farend_ch_;
};

}  // namespace asr_frontend
