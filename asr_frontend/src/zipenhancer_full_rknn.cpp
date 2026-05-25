#include "asr_frontend/zipenhancer_full_rknn.hpp"

#include "asr_frontend/rknn_session.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace asr_frontend {
namespace {

constexpr float kEps = 1e-6f;

}  // namespace

Eigen::VectorXf ZipEnhancerFullRknn::audio_norm_row(const Eigen::Ref<const Eigen::VectorXf>& x) {
  const float rms = std::sqrt((x.array().square().mean()));
  const float scalar = std::pow(10.f, -25.f / 20.f) / (rms + kEps);
  Eigen::VectorXf xx = x * scalar;
  Eigen::VectorXf pow_x = xx.array().square();
  const float avg_pow = pow_x.mean();
  Eigen::ArrayXf mask = (pow_x.array() > avg_pow).cast<float>();
  const float sum_mask = mask.sum();
  float rmsx = rms;
  if (sum_mask > 0.f) {
    rmsx = std::sqrt((pow_x.array() * mask).sum() / sum_mask);
  }
  const float scalarx = std::pow(10.f, -25.f / 20.f) / (rmsx + kEps);
  return xx * scalarx;
}

ZipEnhancerFullRknn::ZipEnhancerFullRknn(const std::string& rknn_path, float decode_window_sec,
                                         int sample_rate)
    : session_(std::make_unique<RknnSession>(rknn_path)),
      decode_window_sec_(decode_window_sec),
      sample_rate_(sample_rate) {
  if (session_->io_num().n_input < 1 || session_->io_num().n_output < 1) {
    throw std::runtime_error("ZipEnhancerFullRknn: unexpected IO count");
  }
  input_type_ = session_->input_attrs()[0].type;
  input_fmt_ = session_->input_attrs()[0].fmt;
}

Eigen::MatrixXf ZipEnhancerFullRknn::preprocess(const float* bct, int B, int C, int T) const {
  Eigen::MatrixXf out(B, T);
  for (int b = 0; b < B; ++b) {
    Eigen::VectorXf row(T);
    for (int t = 0; t < T; ++t) {
      float acc = 0.f;
      for (int c = 0; c < C; ++c) {
        acc += bct[((b * C) + c) * T + t];
      }
      row[t] = acc / static_cast<float>(C);
    }
    out.row(b) = audio_norm_row(row).transpose();
  }
  return out;
}

std::vector<Eigen::VectorXf> ZipEnhancerFullRknn::decode_stream(const float* bct, int B, int C,
                                                                 int T, bool is_first,
                                                                 bool is_last) {
  Eigen::MatrixXf ndarray = preprocess(bct, B, C, T);
  const int window = static_cast<int>(static_cast<float>(sample_rate_) * decode_window_sec_);
  const int stride = static_cast<int>(static_cast<float>(window) * 0.75f);
  const int give_up_length = (window - stride) / 2;
  int padding = 0;

  if (is_last && T < window) {
    padding = window - T;
    Eigen::MatrixXf padded(B, window);
    padded.leftCols(T) = ndarray;
    padded.rightCols(padding).setZero();
    ndarray = std::move(padded);
    T = window;
  } else if (!is_last && T != window) {
    if (T > window) {
      ndarray = ndarray.leftCols(window);
      T = window;
    } else if (T < window) {
      padding = window - T;
      Eigen::MatrixXf padded(B, window);
      padded.leftCols(T) = ndarray;
      padded.rightCols(padding).setZero();
      ndarray = std::move(padded);
      T = window;
    }
  }

  if (T != window) {
    throw std::runtime_error("ZipEnhancerFullRknn: input length must equal fixed RKNN window");
  }

  std::vector<float> input_flat(static_cast<size_t>(B * T));
  for (int b = 0; b < B; ++b) {
    for (int t = 0; t < T; ++t) {
      input_flat[static_cast<size_t>(b * T + t)] = ndarray(b, t);
    }
  }

  rknn_input in{};
  in.index = 0;
  in.buf = input_flat.data();
  in.size = static_cast<uint32_t>(input_flat.size() * sizeof(float));
  in.type = input_type_;
  in.fmt = input_fmt_;
  in.pass_through = 0;

  std::vector<rknn_output> outputs(session_->io_num().n_output);
  for (uint32_t i = 0; i < session_->io_num().n_output; ++i) {
    outputs[i].index = i;
    outputs[i].want_float = 1;
    outputs[i].is_prealloc = 0;
  }

  session_->run({in}, outputs);

  const float* out_ptr = reinterpret_cast<const float*>(outputs[0].buf);
  const rknn_tensor_attr& out_attr = session_->output_attrs()[0];
  int out_T = static_cast<int>(out_attr.n_elems / std::max(1, B));
  if (out_attr.n_dims >= 2) {
    out_T = static_cast<int>(out_attr.dims[out_attr.n_dims - 1]);
  }

  std::vector<Eigen::VectorXf> batch_out;
  batch_out.reserve(B);
  for (int b = 0; b < B; ++b) {
    Eigen::VectorXf tmp(out_T);
    for (int t = 0; t < out_T; ++t) {
      tmp[t] = out_ptr[static_cast<size_t>(b * out_T + t)];
    }
    Eigen::VectorXf cut;
    if (is_first) {
      const int n = std::max(0, static_cast<int>(tmp.size()) - give_up_length);
      cut = tmp.head(n);
    } else if (is_last) {
      const int n = std::max(0, static_cast<int>(tmp.size()) - give_up_length);
      cut = tmp.tail(n);
    } else {
      const int mid = static_cast<int>(tmp.size()) - 2 * give_up_length;
      cut = tmp.segment(give_up_length, std::max(0, mid));
    }
    for (Eigen::Index i = 0; i < cut.size(); ++i) {
      float v = cut[i];
      if (std::isnan(v)) {
        v = 0.f;
      } else if (v > 1.f) {
        v = 1.f;
      } else if (v < -1.f) {
        v = -1.f;
      }
      cut[i] = v;
    }
    if (padding > 0 && cut.size() >= padding) {
      cut.conservativeResize(cut.size() - padding);
    }
    batch_out.push_back(std::move(cut));
  }

  rknn_outputs_release(session_->ctx(), static_cast<uint32_t>(outputs.size()), outputs.data());
  return batch_out;
}

}  // namespace asr_frontend
