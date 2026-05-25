#pragma once

#include <Eigen/Dense>
#include <cassert>
#include <optional>

namespace asr_frontend {

/// Ring buffer for shape (channels, time), aligned with Python
/// `StreamingRingBuffer2D`.
class StreamingRingBuffer2D {
 public:
  /// `decode_window` is in seconds (same as Python `args.decode_window`).
  StreamingRingBuffer2D(int sampling_rate, float decode_window, int channels,
                        std::optional<int> stride_override = std::nullopt,
                        std::optional<int> capacity_override = std::nullopt);

  int window() const { return window_; }
  int stride() const { return stride_; }
  int capacity() const { return capacity_; }
  int channels() const { return channels_; }
  int size() const { return size_; }

  void push(const Eigen::Ref<const Eigen::MatrixXf>& chunk);

  bool ready() const { return size_ >= window_; }

  /// Returns a copy (channels, window).
  Eigen::MatrixXf get_next_window();

  /// Returns remaining (channels, samples) or nullopt if empty.
  std::optional<Eigen::MatrixXf> flush();

 private:
  int window_{};
  int stride_{};
  int capacity_{};
  int channels_{};

  Eigen::MatrixXf buffer_;
  int write_pos_{0};
  int read_pos_{0};
  int size_{0};
};

}  // namespace asr_frontend
