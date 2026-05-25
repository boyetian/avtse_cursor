#include "asr_frontend/stream_inference.hpp"

#include "asr_frontend/audio_resampler.hpp"

#include <algorithm>
#include <cstring>
#include <stdexcept>

namespace asr_frontend {

StreamInference::StreamInference(const StreamInferenceOptions& opt) : opt_(opt) {
  const int sr =
      opt_.enable_aec ? DfsmnAecConfig::kSampleRate
                      : (opt_.enable_speech_enhancement ? 16000 : 16000);
  api_ = std::make_unique<AsrFrontendApi>(
      opt_.enable_aec, opt_.enable_speech_enhancement, opt_.paths, opt_.se_decode_window_sec, sr,
      8, 16);
  target_sr_ = api_->target_sr();
  reset_state_();
}

void StreamInference::reset() { reset_state_(); }

void StreamInference::reset_state_() {
  nearend_buffer_.reset();
  farend_buffer_.reset();
  se_buffer_.reset();
  is_first_aec_ = true;
  is_first_se_ = true;
  current_sr_.reset();
  se_ch_ = 1;
  channels_.reset();
  farend_ch_.reset();
}

Eigen::MatrixXf StreamInference::resample_if_needed_(const Eigen::Ref<const Eigen::MatrixXf>& audio,
                                                     int sampling_rate) {
  current_sr_ = sampling_rate;
  if (sampling_rate == target_sr_) {
    return audio;
  }
  return resample_audio(audio, sampling_rate, target_sr_);
}

void StreamInference::init_buffers_(int nearend_ch, int farend_ch) {
  channels_ = nearend_ch;
  farend_ch_ = farend_ch;

  const float input_decode_window =
      opt_.enable_aec ? kAecDecodeWindowSec : opt_.se_decode_window_sec;
  const std::optional<int> input_stride =
      opt_.enable_aec ? std::optional<int>(kAecRingStride) : std::nullopt;

  nearend_buffer_ = std::make_unique<StreamingRingBuffer2D>(
      target_sr_, input_decode_window, nearend_ch, input_stride, std::nullopt);
  farend_buffer_ = std::make_unique<StreamingRingBuffer2D>(
      target_sr_, input_decode_window, farend_ch, input_stride, std::nullopt);

  if (api_->enable_se()) {
    se_ch_ = opt_.se_channels.value_or(1);
    se_buffer_ = std::make_unique<StreamingRingBuffer2D>(
        target_sr_, opt_.se_decode_window_sec, se_ch_, std::nullopt, std::nullopt);
  }
}

Eigen::VectorXf StreamInference::run_aec_(const Eigen::Ref<const Eigen::MatrixXf>& nearend_win,
                                           const Eigen::MatrixXf* farend_win, bool /*is_last*/) {
  if (farend_win != nullptr && api_->enable_aec()) {
    Eigen::VectorXf y = api_->call_aec_stream(nearend_win, *farend_win);
    if (is_first_aec_) {
      is_first_aec_ = false;
    }
    return y;
  }
  return nearend_win.colwise().mean().transpose();
}

std::vector<Eigen::VectorXf> StreamInference::split_batch_output_(
    const std::vector<Eigen::VectorXf>& se_out) {
  return se_out;
}

std::vector<Eigen::VectorXf> StreamInference::run_se_(const Eigen::Ref<const Eigen::VectorXf>& aec_mono,
                                                       bool should_flush) {
  std::vector<Eigen::VectorXf> outputs;
  if (!se_buffer_) {
    outputs.push_back(aec_mono);
    return outputs;
  }

  Eigen::MatrixXf aec_float(se_ch_, aec_mono.size());
  for (int c = 0; c < se_ch_; ++c) {
    aec_float.row(c) = aec_mono.transpose();
  }

  se_buffer_->push(aec_float);

  auto call_zip = [&](const float* data, int B, int C, int T, bool first, bool last) {
    return api_->call_se_stream(data, B, C, T, first, last);
  };

  if (!opt_.enable_batch) {
    while (se_buffer_->ready()) {
      Eigen::MatrixXf se_win = se_buffer_->get_next_window();
      std::vector<float> packed(static_cast<size_t>(1 * se_ch_ * se_win.cols()));
      for (int c = 0; c < se_ch_; ++c) {
        for (int t = 0; t < se_win.cols(); ++t) {
          packed[static_cast<size_t>(c * se_win.cols() + t)] = se_win(c, t);
        }
      }
      auto outs = call_zip(packed.data(), 1, se_ch_, static_cast<int>(se_win.cols()), is_first_se_,
                           false);
      if (is_first_se_) {
        is_first_se_ = false;
      }
      for (auto& o : outs) {
        outputs.push_back(std::move(o));
      }
    }
  } else {
    std::vector<Eigen::MatrixXf> se_wins;
    while (se_buffer_->ready()) {
      se_wins.push_back(se_buffer_->get_next_window());
    }
    if (!se_wins.empty()) {
      const int B = static_cast<int>(se_wins.size());
      const int T = static_cast<int>(se_wins[0].cols());
      std::vector<float> packed(static_cast<size_t>(B * se_ch_ * T));
      for (int b = 0; b < B; ++b) {
        for (int c = 0; c < se_ch_; ++c) {
          for (int t = 0; t < T; ++t) {
            packed[static_cast<size_t>(((b * se_ch_) + c) * T + t)] =
                se_wins[static_cast<size_t>(b)](c, t);
          }
        }
      }
      if (is_first_se_) {
        if (B == 1) {
          auto outs = call_zip(packed.data(), 1, se_ch_, T, true, false);
          is_first_se_ = false;
          for (auto& o : split_batch_output_(outs)) {
            outputs.push_back(std::move(o));
          }
        } else {
          std::vector<float> first_pack(static_cast<size_t>(1 * se_ch_ * T));
          std::memcpy(first_pack.data(), packed.data(), first_pack.size() * sizeof(float));
          auto first_out = call_zip(first_pack.data(), 1, se_ch_, T, true, false);
          is_first_se_ = false;
          for (auto& o : split_batch_output_(first_out)) {
            outputs.push_back(std::move(o));
          }
          std::vector<float> rest_pack(static_cast<size_t>((B - 1) * se_ch_ * T));
          std::memcpy(rest_pack.data(), packed.data() + first_pack.size(),
                       rest_pack.size() * sizeof(float));
          auto rest_out = call_zip(rest_pack.data(), B - 1, se_ch_, T, false, false);
          for (auto& o : split_batch_output_(rest_out)) {
            outputs.push_back(std::move(o));
          }
        }
      } else {
        auto outs = call_zip(packed.data(), B, se_ch_, T, false, false);
        for (auto& o : split_batch_output_(outs)) {
          outputs.push_back(std::move(o));
        }
      }
    }
  }

  if (should_flush && se_buffer_) {
    auto rem = se_buffer_->flush();
    if (rem.has_value() && rem->size() > 0) {
      std::vector<float> packed(static_cast<size_t>(1 * se_ch_ * rem->cols()));
      for (int c = 0; c < se_ch_; ++c) {
        for (int t = 0; t < rem->cols(); ++t) {
          packed[static_cast<size_t>(c * rem->cols() + t)] = (*rem)(c, t);
        }
      }
      auto outs = call_zip(packed.data(), 1, se_ch_, static_cast<int>(rem->cols()), is_first_se_, true);
      if (is_first_se_) {
        is_first_se_ = false;
      }
      for (auto& o : outs) {
        outputs.push_back(std::move(o));
      }
    }
  }

  return outputs;
}

void StreamInference::push_and_process_(const Eigen::Ref<const Eigen::MatrixXf>& nearend,
                                       const std::optional<Eigen::MatrixXf>& farend,
                                       std::vector<Eigen::VectorXf>& outputs) {
  const int n = static_cast<int>(nearend.cols());
  int offset = 0;
  while (offset < n) {
    while (nearend_buffer_->ready()) {
      Eigen::MatrixXf nearend_win = nearend_buffer_->get_next_window();
      const Eigen::MatrixXf* farend_ptr = nullptr;
      Eigen::MatrixXf farend_copy;
      if (farend_buffer_ && farend_buffer_->ready()) {
        farend_copy = farend_buffer_->get_next_window();
        farend_ptr = &farend_copy;
      }
      Eigen::VectorXf aec_out = run_aec_(nearend_win, farend_ptr, false);
      auto se_out = run_se_(aec_out, false);
      for (auto& o : se_out) {
        outputs.push_back(std::move(o));
      }
    }

    const int available = nearend_buffer_->capacity() - nearend_buffer_->size();
    const int chunk_size = std::min(available, n - offset);
    nearend_buffer_->push(nearend.block(0, offset, nearend.rows(), chunk_size));
    if (farend.has_value() && farend->cols() > 0) {
      farend_buffer_->push(farend->block(0, offset, farend->rows(), chunk_size));
    }
    offset += chunk_size;

    while (nearend_buffer_->ready()) {
      Eigen::MatrixXf nearend_win = nearend_buffer_->get_next_window();
      const Eigen::MatrixXf* farend_ptr = nullptr;
      Eigen::MatrixXf farend_copy;
      if (farend_buffer_ && farend_buffer_->ready()) {
        farend_copy = farend_buffer_->get_next_window();
        farend_ptr = &farend_copy;
      }
      Eigen::VectorXf aec_out = run_aec_(nearend_win, farend_ptr, false);
      auto se_out = run_se_(aec_out, false);
      for (auto& o : se_out) {
        outputs.push_back(std::move(o));
      }
    }
  }
}

std::optional<Eigen::VectorXf> StreamInference::flush_aec_() {
  if (!nearend_buffer_) {
    return std::nullopt;
  }
  auto ne = nearend_buffer_->flush();
  if (!ne.has_value() || ne->size() == 0) {
    return std::nullopt;
  }
  const Eigen::MatrixXf* farend_ptr = nullptr;
  Eigen::MatrixXf farend_copy;
  if (farend_buffer_) {
    auto fa = farend_buffer_->flush();
    if (fa.has_value() && fa->size() > 0) {
      farend_copy = *fa;
      farend_ptr = &farend_copy;
    }
  }
  return run_aec_(*ne, farend_ptr, true);
}

std::vector<Eigen::VectorXf> StreamInference::stream_inference(
    const std::optional<Eigen::MatrixXf>& nearend, const std::optional<Eigen::MatrixXf>& farend,
    bool is_start, bool is_end, bool flush_buffer, int sampling_rate) {
  std::vector<Eigen::VectorXf> outputs;

  if (is_start) {
    reset_state_();
  }

  if (nearend.has_value() && nearend->size() > 0 && nearend->cols() > 0) {
    Eigen::MatrixXf ne = resample_if_needed_(*nearend, sampling_rate);
    std::optional<Eigen::MatrixXf> fe;
    if (farend.has_value() && farend->size() > 0 && farend->cols() > 0) {
      fe = resample_if_needed_(*farend, sampling_rate);
    }

    if (!nearend_buffer_) {
      const int ne_ch = opt_.nearend_channels.value_or(static_cast<int>(ne.rows()));
      const int fa_ch =
          opt_.farend_channels.value_or((fe.has_value() && fe->cols() > 0)
                                            ? static_cast<int>(fe->rows())
                                            : ne_ch);
      init_buffers_(ne_ch, fa_ch);
    }

    push_and_process_(ne, fe, outputs);
  }

  if (is_end || flush_buffer) {
    auto aec_vec = flush_aec_();
    if (aec_vec.has_value() && aec_vec->size() > 0) {
      auto tail = run_se_(*aec_vec, true);
      for (auto& o : tail) {
        outputs.push_back(std::move(o));
      }
    }
  }

  if (is_end) {
    reset_state_();
  }

  return outputs;
}

}  // namespace asr_frontend
