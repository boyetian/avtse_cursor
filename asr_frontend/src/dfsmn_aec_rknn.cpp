#include "asr_frontend/dfsmn_aec_rknn.hpp"

#include "asr_frontend/rknn_session.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace asr_frontend {

static Eigen::VectorXf mean_channels(const Eigen::Ref<const Eigen::MatrixXf>& x) {
  if (x.rows() == 1) {
    return x.row(0).transpose();
  }
  return x.colwise().mean().transpose();
}

DfsmnAecRknn::DfsmnAecRknn(const std::string& rknn_path) : session_(std::make_unique<RknnSession>(rknn_path)) {
  if (session_->io_num().n_input < 2 || session_->io_num().n_output < 1) {
    throw std::runtime_error("DfsmnAecRknn: unexpected IO count");
  }
  input_type_ = session_->input_attrs()[0].type;
  input_fmt_ = session_->input_attrs()[0].fmt;
}

Eigen::Matrix<std::int16_t, Eigen::Dynamic, 1> DfsmnAecRknn::decode_stream(
    const Eigen::Ref<const Eigen::MatrixXf>& nearend,
    const Eigen::Ref<const Eigen::MatrixXf>& farend) {
  Eigen::VectorXf ne = mean_channels(nearend);
  Eigen::VectorXf fa = mean_channels(farend);
  if (ne.size() != fa.size()) {
    throw std::runtime_error("DfsmnAecRknn: nearend/farend time length mismatch");
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

  // RKNN model uses float inputs [1, 1, kChunkSize] after ONNX patch for toolkit.
  std::vector<float> ne_f(kChunkSize);
  std::vector<float> fa_f(kChunkSize);
  for (int i = 0; i < kChunkSize; ++i) {
    ne_f[static_cast<size_t>(i)] = ne[i];
    fa_f[static_cast<size_t>(i)] = fa[i];
  }

  std::vector<rknn_input> inputs(2);
  const size_t bytes = static_cast<size_t>(kChunkSize) * sizeof(float);
  for (uint32_t idx = 0; idx < 2; ++idx) {
    rknn_input in{};
    in.index = idx;
    in.type = RKNN_TENSOR_FLOAT32;
    in.fmt = input_fmt_;
    in.pass_through = 0;
    in.buf = (idx == 0) ? ne_f.data() : fa_f.data();
    in.size = static_cast<uint32_t>(bytes);
    inputs[idx] = in;
  }

  std::vector<rknn_output> outputs(session_->io_num().n_output);
  for (uint32_t i = 0; i < session_->io_num().n_output; ++i) {
    outputs[i].index = i;
    outputs[i].want_float = 0;
    outputs[i].is_prealloc = 0;
  }

  session_->run(inputs, outputs);

  const rknn_tensor_attr& out_attr = session_->output_attrs()[0];
  const size_t out_elems = out_attr.n_elems;
  std::vector<std::int16_t> clipped(out_elems);

  if (out_attr.type == RKNN_TENSOR_INT16) {
    const auto* out_i16 = reinterpret_cast<const std::int16_t*>(outputs[0].buf);
    for (size_t i = 0; i < out_elems; ++i) {
      clipped[i] = static_cast<std::int16_t>(std::clamp<int>(out_i16[i], -32768, 32767));
    }
  } else {
    const auto* out_f = reinterpret_cast<const float*>(outputs[0].buf);
    for (size_t i = 0; i < out_elems; ++i) {
      clipped[i] = static_cast<std::int16_t>(
          std::lrintf(std::clamp(out_f[i], -32768.f, 32767.f)));
    }
  }

  rknn_outputs_release(session_->ctx(), static_cast<uint32_t>(outputs.size()), outputs.data());

  const int out_len = static_cast<int>(std::min<Eigen::Index>(
      static_cast<Eigen::Index>(out_elems), static_cast<Eigen::Index>(total_len)));
  Eigen::Matrix<std::int16_t, Eigen::Dynamic, 1> ret(out_len);
  for (int i = 0; i < out_len; ++i) {
    ret[i] = clipped[static_cast<size_t>(i)];
  }
  return ret;
}

}  // namespace asr_frontend
