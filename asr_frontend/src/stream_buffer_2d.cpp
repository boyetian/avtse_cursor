#include "asr_frontend/stream_buffer_2d.hpp"

#include <algorithm>

namespace asr_frontend {

StreamingRingBuffer2D::StreamingRingBuffer2D(int sampling_rate, float decode_window,
                                            int channels,
                                            std::optional<int> stride_override,
                                            std::optional<int> capacity_override)
    : channels_(channels) {
  window_ = static_cast<int>(static_cast<float>(sampling_rate) * decode_window);
  if (stride_override.has_value()) {
    stride_ = *stride_override;
  } else {
    stride_ = static_cast<int>(static_cast<float>(window_) * 0.75f);
  }
  capacity_ = capacity_override.value_or(window_ * 4);
  assert(capacity_ >= window_ + stride_);
  buffer_.resize(channels_, capacity_);
  buffer_.setZero();
}

void StreamingRingBuffer2D::push(const Eigen::Ref<const Eigen::MatrixXf>& chunk) {
  assert(chunk.rows() == channels_);
  const int n = static_cast<int>(chunk.cols());
  if (n >= capacity_) {
    buffer_ = chunk.rightCols(capacity_);
    write_pos_ = 0;
    read_pos_ = 0;
    size_ = capacity_;
    return;
  }

  const int end = write_pos_ + n;
  if (end <= capacity_) {
    buffer_.middleCols(write_pos_, n) = chunk;
  } else {
    const int first = capacity_ - write_pos_;
    buffer_.middleCols(write_pos_, first) = chunk.leftCols(first);
    buffer_.leftCols(end % capacity_) = chunk.rightCols(n - first);
  }

  write_pos_ = end % capacity_;
  size_ = std::min(size_ + n, capacity_);
  if (size_ == capacity_) {
    read_pos_ = (write_pos_ + stride_) % capacity_;
    size_ -= stride_;
  }
}

Eigen::MatrixXf StreamingRingBuffer2D::get_next_window() {
  assert(ready());
  const int start = read_pos_;
  const int end = start + window_;
  Eigen::MatrixXf out(channels_, window_);
  if (end <= capacity_) {
    out = buffer_.middleCols(start, window_);
  } else {
    const int first = capacity_ - start;
    out.leftCols(first) = buffer_.rightCols(first);
    out.rightCols(window_ - first) = buffer_.leftCols(end % capacity_);
  }
  read_pos_ = (read_pos_ + stride_) % capacity_;
  size_ -= stride_;
  return out;
}

std::optional<Eigen::MatrixXf> StreamingRingBuffer2D::flush() {
  if (size_ <= 0) {
    return std::nullopt;
  }
  const int start = read_pos_;
  const int end = start + size_;
  Eigen::MatrixXf data(channels_, size_);
  if (end <= capacity_) {
    data = buffer_.middleCols(start, size_);
  } else {
    const int first = capacity_ - start;
    data.leftCols(first) = buffer_.rightCols(first);
    data.rightCols(size_ - first) = buffer_.leftCols(end % capacity_);
  }
  size_ = 0;
  return data;
}

}  // namespace asr_frontend
