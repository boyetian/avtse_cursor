#pragma once

#include <Eigen/Dense>

namespace asr_frontend {

/// Band-limited resampling via libsoxr (SOXR_LSR2Q fast sinc, float I/O).
Eigen::MatrixXf resample_audio(const Eigen::Ref<const Eigen::MatrixXf>& input,
                               int input_sr, int output_sr);

}  // namespace asr_frontend
