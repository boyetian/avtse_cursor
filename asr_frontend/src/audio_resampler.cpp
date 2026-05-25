#include "asr_frontend/audio_resampler.hpp"

#include <soxr.h>

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

namespace asr_frontend {

namespace {

static size_t estimate_output_samples(size_t input_samples, int input_sr, int output_sr) {
  if (input_samples == 0) {
    return 0;
  }
  const double ratio = static_cast<double>(output_sr) / static_cast<double>(input_sr);
  // Extra headroom for anti-alias filter delay / rounding.
  return static_cast<size_t>(std::ceil(static_cast<double>(input_samples) * ratio)) + 64;
}

static Eigen::VectorXf resample_channel_soxr(const float* samples, size_t num_samples,
                                               int input_sr, int output_sr) {
  if (num_samples == 0) {
    return Eigen::VectorXf(0);
  }
  if (input_sr == output_sr) {
    Eigen::VectorXf y(static_cast<Eigen::Index>(num_samples));
    for (size_t i = 0; i < num_samples; ++i) {
      y(static_cast<Eigen::Index>(i)) = samples[i];
    }
    return y;
  }

  const size_t olen = estimate_output_samples(num_samples, input_sr, output_sr);
  std::vector<float> out_buf(olen);

  size_t idone = 0;
  size_t odone = 0;

  const soxr_io_spec_t io_spec = soxr_io_spec(SOXR_FLOAT32_I, SOXR_FLOAT32_I);
  const soxr_quality_spec_t q_spec = soxr_quality_spec(SOXR_LSR2Q, 0);

  const soxr_error_t err = soxr_oneshot(
      static_cast<double>(input_sr), static_cast<double>(output_sr), 1u,
      static_cast<soxr_in_t>(static_cast<void const*>(samples)), num_samples, &idone,
      static_cast<soxr_out_t>(out_buf.data()), olen, &odone, &io_spec, &q_spec, nullptr);

  if (err != nullptr) {
    throw std::runtime_error(std::string("soxr_oneshot: ") + soxr_strerror(err));
  }
  if (idone != num_samples) {
    throw std::runtime_error("soxr_oneshot: did not consume entire input buffer");
  }

  Eigen::VectorXf y(static_cast<Eigen::Index>(odone));
  for (size_t i = 0; i < odone; ++i) {
    y(static_cast<Eigen::Index>(i)) = out_buf[i];
  }
  return y;
}

}  // namespace

Eigen::MatrixXf resample_audio(const Eigen::Ref<const Eigen::MatrixXf>& input, int input_sr,
                               int output_sr) {
  if (input_sr <= 0 || output_sr <= 0) {
    throw std::invalid_argument("resample_audio: sample rates must be positive");
  }
  if (input_sr == output_sr) {
    return input;
  }

  const int ch = static_cast<int>(input.rows());
  const Eigen::Index cols = input.cols();
  if (ch == 0) {
    return input;
  }

  Eigen::MatrixXf out(ch, 0);
  for (int c = 0; c < ch; ++c) {
    std::vector<float> ch_in(static_cast<size_t>(cols));
    for (Eigen::Index i = 0; i < cols; ++i) {
      ch_in[static_cast<size_t>(i)] = input(c, i);
    }
    Eigen::VectorXf row = resample_channel_soxr(ch_in.data(), ch_in.size(), input_sr, output_sr);
    if (c == 0) {
      out.resize(ch, row.size());
    } else if (row.size() != out.cols()) {
      throw std::runtime_error("resample_audio: channel length mismatch");
    }
    out.row(c) = row.transpose();
  }
  return out;
}

}  // namespace asr_frontend
