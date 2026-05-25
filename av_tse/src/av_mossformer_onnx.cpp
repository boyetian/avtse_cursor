#include "av_tse/av_mossformer_onnx.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace av_tse {

namespace {

Eigen::VectorXf squeezeOutput(const float* data, size_t count) {
  Eigen::VectorXf out(static_cast<Eigen::Index>(count));
  for (size_t i = 0; i < count; ++i) {
    out(static_cast<Eigen::Index>(i)) = data[i];
  }
  return out;
}

}  // namespace

AvMossformerOnnx::AvMossformerOnnx(const std::string& onnx_path, int num_threads)
    : env_(ORT_LOGGING_LEVEL_WARNING, "av_tse") {
  session_options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
  if (num_threads > 0) {
    session_options_.SetIntraOpNumThreads(num_threads);
    session_options_.SetInterOpNumThreads(1);
  }
#ifdef _WIN32
  std::wstring wpath(onnx_path.begin(), onnx_path.end());
  session_ = std::make_unique<Ort::Session>(env_, wpath.c_str(), session_options_);
#else
  session_ = std::make_unique<Ort::Session>(env_, onnx_path.c_str(), session_options_);
#endif

  const size_t n_in = session_->GetInputCount();
  const size_t n_out = session_->GetOutputCount();
  if (n_in < 2 || n_out < 1) {
    throw std::runtime_error("AvMossformerOnnx: expected >=2 inputs");
  }
  owned_input_names_.reserve(n_in);
  for (size_t i = 0; i < n_in; ++i) {
    Ort::AllocatedStringPtr name = session_->GetInputNameAllocated(i, allocator_);
    owned_input_names_.emplace_back(name.get());
  }
  input_names_.push_back("mixture");
  input_names_.push_back("ref");
  for (const auto& n : owned_input_names_) {
    if (n != "mixture" && n != "ref") {
      throw std::runtime_error("unexpected ONNX input: " + n);
    }
  }
  for (size_t i = 0; i < n_out; ++i) {
    Ort::AllocatedStringPtr name = session_->GetOutputNameAllocated(i, allocator_);
    owned_output_names_.emplace_back(name.get());
    output_names_.push_back(owned_output_names_.back().c_str());
  }
}

Eigen::VectorXf AvMossformerOnnx::run(const Eigen::Ref<const Eigen::VectorXf>& mixture,
                                      const std::vector<cv::Mat>& ref_frames) {
  const int T = static_cast<int>(mixture.size());
  const int Tv = static_cast<int>(ref_frames.size());
  if (T <= 0 || Tv <= 0) {
    return Eigen::VectorXf(0);
  }
  const int H = ref_frames[0].rows;
  const int W = ref_frames[0].cols;
  const int C = ref_frames[0].channels();

  std::vector<float> mix_flat(static_cast<size_t>(T));
  for (int t = 0; t < T; ++t) {
    mix_flat[static_cast<size_t>(t)] = mixture[t];
  }

  std::vector<float> ref_flat(static_cast<size_t>(Tv) * static_cast<size_t>(H) * static_cast<size_t>(W) *
                            static_cast<size_t>(C));
  for (int t = 0; t < Tv; ++t) {
    const cv::Mat& f = ref_frames[static_cast<size_t>(t)];
    if (f.rows != H || f.cols != W || f.channels() != C) {
      throw std::runtime_error("ref frame shape mismatch");
    }
    for (int y = 0; y < H; ++y) {
      for (int x = 0; x < W; ++x) {
        const cv::Vec3f* row = f.ptr<cv::Vec3f>(y);
        for (int c = 0; c < C; ++c) {
          ref_flat[static_cast<size_t>(((t * H + y) * W + x) * C + c)] = row[x][c];
        }
      }
    }
  }

  const std::array<int64_t, 2> mix_shape{1, T};
  const std::array<int64_t, 5> ref_shape{1, Tv, H, W, C};

  Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  Ort::Value mix_tensor =
      Ort::Value::CreateTensor<float>(mem, mix_flat.data(), mix_flat.size(), mix_shape.data(), 2);
  Ort::Value ref_tensor =
      Ort::Value::CreateTensor<float>(mem, ref_flat.data(), ref_flat.size(), ref_shape.data(), 5);

  std::array<Ort::Value, 2> inputs{std::move(mix_tensor), std::move(ref_tensor)};
  auto outputs = session_->Run(Ort::RunOptions{nullptr}, input_names_.data(), inputs.data(), 2,
                               output_names_.data(), 1);

  const auto& out_info = outputs[0].GetTensorTypeAndShapeInfo();
  const float* out_ptr = outputs[0].GetTensorData<float>();
  const auto out_shape = out_info.GetShape();
  int out_t = T;
  if (out_shape.size() >= 2) {
    const int64_t last = out_shape.back();
    if (last > 0) {
      out_t = static_cast<int>(last);
    }
  } else {
    out_t = static_cast<int>(out_info.GetElementCount());
  }
  out_t = std::min(out_t, T);
  Eigen::VectorXf y(out_t);
  for (int i = 0; i < out_t; ++i) {
    y[i] = out_ptr[i];
  }
  if (out_t < T) {
    Eigen::VectorXf padded(T);
    padded.setZero();
    padded.head(out_t) = y;
    return padded;
  }
  return y;
}

}  // namespace av_tse
