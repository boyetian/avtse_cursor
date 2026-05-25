#include "asr_frontend/dfsmn_aec_onnx.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace asr_frontend {

static Eigen::VectorXf mean_channels(const Eigen::Ref<const Eigen::MatrixXf>& x) {
  if (x.rows() == 1) {
    return x.row(0).transpose();
  }
  return x.colwise().mean().transpose();
}

DfsmnAecOnnx::DfsmnAecOnnx(const std::string& onnx_path, int intra_op_threads)
    : env_(ORT_LOGGING_LEVEL_WARNING, "asr_frontend"), session_options_{}, session_{} {
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

  const size_t n_in = session_->GetInputCount();
  const size_t n_out = session_->GetOutputCount();
  if (n_in < 2 || n_out < 1) {
    throw std::runtime_error("DfsmnAecOnnx: unexpected IO count");
  }
  // Reserve so emplace_back never reallocates while we store c_str() into input_names_.
  owned_input_names_.reserve(n_in);
  input_names_.reserve(n_in);
  for (size_t i = 0; i < n_in; ++i) {
    Ort::AllocatedStringPtr name = session_->GetInputNameAllocated(i, allocator_);
    owned_input_names_.emplace_back(name.get());
    input_names_.push_back(owned_input_names_.back().c_str());
  }
  Ort::AllocatedStringPtr on = session_->GetOutputNameAllocated(0, allocator_);
  owned_output_names_.emplace_back(on.get());
  output_names_.push_back(owned_output_names_.back().c_str());
}

Eigen::Matrix<std::int16_t, Eigen::Dynamic, 1> DfsmnAecOnnx::decode_stream(
    const Eigen::Ref<const Eigen::MatrixXf>& nearend,
    const Eigen::Ref<const Eigen::MatrixXf>& farend) {
  Eigen::VectorXf ne = mean_channels(nearend);
  Eigen::VectorXf fa = mean_channels(farend);
  if (ne.size() != fa.size()) {
    throw std::runtime_error("DfsmnAecOnnx: nearend/farend time length mismatch");
  }
  const Eigen::Index total_len = ne.size();

  if (ne.size() < kChunkSize) {
    Eigen::VectorXf nep = Eigen::VectorXf::Zero(kChunkSize);
    Eigen::VectorXf fap = Eigen::VectorXf::Zero(kChunkSize);
    nep.head(ne.size()) = ne;
    fap.head(fa.size()) = fa;
    ne = std::move(nep);
    fa = std::move(fap);
  } else if (ne.size() > kChunkSize) {
    ne = ne.head(kChunkSize);
    fa = fa.head(kChunkSize);
  }

  Ort::TypeInfo t0 = session_->GetInputTypeInfo(0);
  std::vector<std::int64_t> in_shape = t0.GetTensorTypeAndShapeInfo().GetShape();
  std::vector<std::int64_t> dims;
  dims.reserve(in_shape.size());
  for (size_t i = 0; i < in_shape.size(); ++i) {
    const std::int64_t d = in_shape[i];
    if (d > 0) {
      dims.push_back(d);
    } else if (i + 1 == in_shape.size()) {
      dims.push_back(kChunkSize);
    } else {
      dims.push_back(1);
    }
  }
  int64_t flat =
      std::accumulate(dims.begin(), dims.end(), int64_t{1}, std::multiplies<int64_t>());
  if (flat != kChunkSize) {
    dims = {1, kChunkSize};
  }

  Eigen::Matrix<std::int16_t, Eigen::Dynamic, 1> ne_i16(kChunkSize);
  Eigen::Matrix<std::int16_t, Eigen::Dynamic, 1> fa_i16(kChunkSize);
  for (int i = 0; i < kChunkSize; ++i) {
    ne_i16[i] = static_cast<std::int16_t>(
        std::lrintf(std::clamp(ne[i] * 32768.0f, -32768.0f, 32767.0f)));
    fa_i16[i] = static_cast<std::int16_t>(
        std::lrintf(std::clamp(fa[i] * 32768.0f, -32768.0f, 32767.0f)));
  }

  Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  std::vector<std::int64_t> dims_ne = dims;
  std::vector<std::int64_t> dims_fa = dims;

  Ort::Value in_ne = Ort::Value::CreateTensor<std::int16_t>(
      mem, ne_i16.data(), static_cast<size_t>(ne_i16.size()), dims_ne.data(), dims_ne.size());
  Ort::Value in_fa = Ort::Value::CreateTensor<std::int16_t>(
      mem, fa_i16.data(), static_cast<size_t>(fa_i16.size()), dims_fa.data(), dims_fa.size());

  std::vector<Ort::Value> inputs;
  inputs.push_back(std::move(in_ne));
  inputs.push_back(std::move(in_fa));

  auto outputs = session_->Run(Ort::RunOptions{nullptr}, input_names_.data(), inputs.data(),
                                 inputs.size(), output_names_.data(), 1);

  auto out_info = outputs[0].GetTensorTypeAndShapeInfo();
  const size_t out_elems = static_cast<size_t>(out_info.GetElementCount());
  ONNXTensorElementDataType et = out_info.GetElementType();
  if (et != ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16) {
    throw std::runtime_error("DfsmnAecOnnx: unexpected output element type (expected int16)");
  }
  const std::int16_t* out_i16 = outputs[0].GetTensorData<std::int16_t>();
  std::vector<std::int16_t> clipped(out_elems);
  for (size_t i = 0; i < out_elems; ++i) {
    clipped[i] = static_cast<std::int16_t>(std::clamp<int>(out_i16[i], -32768, 32767));
  }

  const int out_len = static_cast<int>(std::min<Eigen::Index>(
      static_cast<Eigen::Index>(out_elems), static_cast<Eigen::Index>(total_len)));
  Eigen::Matrix<std::int16_t, Eigen::Dynamic, 1> ret(out_len);
  for (int i = 0; i < out_len; ++i) {
    ret[i] = clipped[static_cast<size_t>(i)];
  }
  return ret;
}

}  // namespace asr_frontend
