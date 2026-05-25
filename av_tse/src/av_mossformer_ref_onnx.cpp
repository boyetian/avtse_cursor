#include "av_tse/av_mossformer_ref_onnx.hpp"

#include <algorithm>
#include <stdexcept>

namespace av_tse {

namespace {

int dimAt(const Ort::Session& session, size_t input_idx, int dim_idx, int fallback = 1) {
  Ort::TypeInfo ti = session.GetInputTypeInfo(input_idx);
  auto tensor_info = ti.GetTensorTypeAndShapeInfo();
  const auto shape = tensor_info.GetShape();
  if (dim_idx < 0 || static_cast<size_t>(dim_idx) >= shape.size()) {
    return fallback;
  }
  const int64_t v = shape[static_cast<size_t>(dim_idx)];
  return v > 0 ? static_cast<int>(v) : fallback;
}

}  // namespace

AvMossformerRefOnnx::AvMossformerRefOnnx(const std::string& onnx_path, int num_threads)
    : env_(ORT_LOGGING_LEVEL_WARNING, "av_tse_ref") {
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
  if (n_in < 1) {
    throw std::runtime_error("AvMossformerRefOnnx: expected >=1 input");
  }
  for (size_t i = 0; i < n_in; ++i) {
    Ort::AllocatedStringPtr name = session_->GetInputNameAllocated(i, allocator_);
    owned_input_names_.emplace_back(name.get());
    input_names_.push_back(owned_input_names_.back().c_str());
  }
  for (size_t i = 0; i < session_->GetOutputCount(); ++i) {
    Ort::AllocatedStringPtr name = session_->GetOutputNameAllocated(i, allocator_);
    owned_output_names_.emplace_back(name.get());
    output_names_.push_back(owned_output_names_.back().c_str());
  }

  ref_frames_ = dimAt(*session_, 0, 1, 18);
  image_size_ = dimAt(*session_, 0, 2, 96);
  const int w = dimAt(*session_, 0, 3, image_size_);
  if (w > 0) {
    image_size_ = w;
  }
}

std::vector<float> AvMossformerRefOnnx::runGrayFrames(const std::vector<cv::Mat>& gray_frames,
                                                      int image_size) {
  const int Tv = static_cast<int>(gray_frames.size());
  if (Tv <= 0) {
    return {};
  }
  const int H = image_size > 0 ? image_size : image_size_;
  const int W = H;

  std::vector<float> gray_flat(static_cast<size_t>(ref_frames_) * static_cast<size_t>(H) *
                               static_cast<size_t>(W), 0.f);
  const int copy_tv = std::min(Tv, ref_frames_);
  for (int t = 0; t < copy_tv; ++t) {
    cv::Mat f = gray_frames[static_cast<size_t>(t)];
    if (f.channels() != 1) {
      throw std::runtime_error("ref gray frame must be single channel");
    }
    cv::Mat resized;
    if (f.rows != H || f.cols != W) {
      cv::resize(f, resized, cv::Size(W, H), 0, 0, cv::INTER_AREA);
      f = resized;
    }
    cv::Mat f32;
    if (f.type() == CV_32FC1) {
      f32 = f;
    } else {
      f.convertTo(f32, CV_32F, 1.0 / 255.0);
    }
    for (int y = 0; y < H; ++y) {
      const float* row = f32.ptr<float>(y);
      for (int x = 0; x < W; ++x) {
        gray_flat[static_cast<size_t>((t * H + y) * W + x)] = row[x];
      }
    }
  }

  const std::array<int64_t, 4> shape{1, ref_frames_, H, W};
  Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  Ort::Value in_tensor =
      Ort::Value::CreateTensor<float>(mem, gray_flat.data(), gray_flat.size(), shape.data(), 4);

  auto outputs = session_->Run(Ort::RunOptions{nullptr}, input_names_.data(), &in_tensor, 1,
                               output_names_.data(), 1);

  const auto& out_info = outputs[0].GetTensorTypeAndShapeInfo();
  const float* out_ptr = outputs[0].GetTensorData<float>();
  const size_t n = out_info.GetElementCount();
  ref_feat_channels_ = dimAt(*session_, 0, 0, 96);
  const auto out_shape = out_info.GetShape();
  if (out_shape.size() >= 2 && out_shape[1] > 0) {
    ref_feat_channels_ = static_cast<int>(out_shape[1]);
  }
  return std::vector<float>(out_ptr, out_ptr + n);
}

}  // namespace av_tse
