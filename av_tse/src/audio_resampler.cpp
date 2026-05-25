#include "av_tse/audio_resampler.hpp"

#include <soxr.h>

#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace av_tse {

namespace {

size_t estimateOutputSamples(size_t input_samples, int input_sr, int output_sr) {
  if (input_samples == 0) {
    return 0;
  }
  const double ratio = static_cast<double>(output_sr) / static_cast<double>(input_sr);
  return static_cast<size_t>(std::ceil(static_cast<double>(input_samples) * ratio)) + 64;
}

}  // namespace

Eigen::VectorXf resample_mono(const Eigen::Ref<const Eigen::VectorXf>& input, int input_sr,
                              int output_sr) {
  if (input_sr <= 0 || output_sr <= 0) {
    throw std::invalid_argument("resample_mono: invalid sample rate");
  }
  const size_t num_samples = static_cast<size_t>(input.size());
  if (num_samples == 0) {
    return Eigen::VectorXf(0);
  }
  if (input_sr == output_sr) {
    return input;
  }

  const size_t olen = estimateOutputSamples(num_samples, input_sr, output_sr);
  std::vector<float> out_buf(olen);
  size_t idone = 0;
  size_t odone = 0;

  const soxr_io_spec_t io_spec = soxr_io_spec(SOXR_FLOAT32_I, SOXR_FLOAT32_I);
  const soxr_quality_spec_t q_spec = soxr_quality_spec(SOXR_LSR2Q, 0);
  const soxr_error_t err = soxr_oneshot(
      static_cast<double>(input_sr), static_cast<double>(output_sr), 1u,
      static_cast<soxr_in_t>(static_cast<void const*>(input.data())), num_samples, &idone,
      static_cast<soxr_out_t>(out_buf.data()), olen, &odone, &io_spec, &q_spec, nullptr);

  if (err != nullptr) {
    throw std::runtime_error(std::string("soxr_oneshot: ") + soxr_strerror(err));
  }
  if (idone != num_samples) {
    throw std::runtime_error("soxr_oneshot: did not consume entire input");
  }

  Eigen::VectorXf y(static_cast<Eigen::Index>(odone));
  for (size_t i = 0; i < odone; ++i) {
    y(static_cast<Eigen::Index>(i)) = out_buf[i];
  }
  return y;
}

}  // namespace av_tse
