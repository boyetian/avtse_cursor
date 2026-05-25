#pragma once

#include <Eigen/Dense>

namespace av_tse {

Eigen::VectorXf resample_mono(const Eigen::Ref<const Eigen::VectorXf>& input, int input_sr,
                              int output_sr);

}  // namespace av_tse
