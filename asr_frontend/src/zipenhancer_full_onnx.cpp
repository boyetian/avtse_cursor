#include "asr_frontend/zipenhancer_full_onnx.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace asr_frontend {
namespace {

constexpr float kEps = 1e-6f;

}  // namespace

Eigen::VectorXf ZipEnhancerFullOnnx::audio_norm_row(const Eigen::Ref<const Eigen::VectorXf>& x) {
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

ZipEnhancerFullOnnx::ZipEnhancerFullOnnx(const std::string& onnx_path, float decode_window_sec,
                                         int sample_rate, int intra_op_threads)
    : env_(ORT_LOGGING_LEVEL_WARNING, "asr_frontend"),
      session_options_{},
      decode_window_sec_(decode_window_sec),
      sample_rate_(sample_rate) {
  session_options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
  session_options_.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
  session_options_.SetIntraOpNumThreads(intra_op_threads);
  session_options_.SetInterOpNumThreads(1);
#ifdef _WIN32
  std::wstring wpath(onnx_path.begin(), onnx_path.end());
  session_ = std::make_unique<Ort::Session>(env_, wpath.c_str(), session_options_);
#else
  session_ = std::make_unique<Ort::Session>(env_, onnx_path.c_str(), session_options_);
#endif

  if (session_->GetInputCount() < 1 || session_->GetOutputCount() < 1) {
    throw std::runtime_error("ZipEnhancerFullOnnx: unexpected IO count");
  }
  Ort::AllocatedStringPtr in0 = session_->GetInputNameAllocated(0, allocator_);
  owned_input_names_.emplace_back(in0.get());
  input_names_.push_back(owned_input_names_.back().c_str());
  Ort::AllocatedStringPtr on0 = session_->GetOutputNameAllocated(0, allocator_);
  owned_output_names_.emplace_back(on0.get());
  output_names_.push_back(owned_output_names_.back().c_str());
}

Eigen::MatrixXf ZipEnhancerFullOnnx::preprocess(const float* bct, int B, int C, int T) const {
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

std::vector<Eigen::VectorXf> ZipEnhancerFullOnnx::decode_stream(const float* bct, int B, int C,
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

  Ort::TypeInfo ti = session_->GetInputTypeInfo(0);
  auto shape = ti.GetTensorTypeAndShapeInfo().GetShape();
  std::vector<std::int64_t> dims;
  dims.reserve(shape.size());
  for (size_t i = 0; i < shape.size(); ++i) {
    const std::int64_t d = shape[i];
    if (d > 0) {
      dims.push_back(d);
    } else if (shape.size() == 2) {
      dims.push_back(i == 0 ? static_cast<std::int64_t>(B) : static_cast<std::int64_t>(T));
    } else {
      dims.push_back(static_cast<std::int64_t>(B * T));
    }
  }
  if (dims.size() == 2) {
    if (dims[0] * dims[1] != B * T) {
      dims[0] = B;
      dims[1] = T;
    }
  }

  std::vector<float> input_flat(static_cast<size_t>(B * T));
  for (int b = 0; b < B; ++b) {
    for (int t = 0; t < T; ++t) {
      input_flat[static_cast<size_t>(b * T + t)] = ndarray(b, t);
    }
  }

  Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  Ort::Value in_tensor = Ort::Value::CreateTensor<float>(
      mem, input_flat.data(), input_flat.size(), dims.data(), dims.size());

  auto outputs = session_->Run(Ort::RunOptions{nullptr}, input_names_.data(), &in_tensor, 1,
                               output_names_.data(), 1);

  auto out_info = outputs[0].GetTensorTypeAndShapeInfo();
  const float* out_ptr = outputs[0].GetTensorData<float>();
  const auto out_shape = out_info.GetShape();
  int out_T = 0;
  if (out_shape.size() >= 2) {
    out_T = static_cast<int>(out_shape.back());
  } else {
    out_T = static_cast<int>(out_info.GetElementCount() / std::max(1, B));
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
  return batch_out;
}

}  // namespace asr_frontend
