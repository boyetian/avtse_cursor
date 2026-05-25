#include "av_tse/av_mossformer_rknn.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace av_tse {

namespace {

int dimAt(const rknn_tensor_attr& attr, int idx, int fallback = 1) {
  if (idx < 0 || idx >= static_cast<int>(attr.n_dims)) {
    return fallback;
  }
  const int v = attr.dims[idx];
  return v > 0 ? v : fallback;
}

}  // namespace

AvMossformerRknn::AvMossformerRknn(const std::string& rknn_path, int audio_len, int ref_frames,
                                   int image_size)
    : session_(std::make_unique<asr_frontend::RknnSession>(rknn_path)), image_size_(image_size) {
  if (session_->io_num().n_input < 2 || session_->io_num().n_output < 1) {
    throw std::runtime_error("AvMossformerRknn: expected >=2 inputs and >=1 output");
  }

  const rknn_tensor_attr& mix_attr = session_->input_attrs()[0];
  const rknn_tensor_attr& ref_attr = session_->input_attrs()[1];

  audio_len_ = audio_len > 0 ? audio_len : dimAt(mix_attr, 1, 9600);

  if (ref_attr.n_dims >= 5) {
    throw std::runtime_error(
        "AvMossformerRknn: 5D RGB ref input not supported. Use AvMossformerSplit "
        "(ORT ref_encoder + RKNN sep with ref_feat).");
  }

  ref_feat_channels_ = dimAt(ref_attr, 1, 96);
  ref_frames_ = ref_frames > 0 ? ref_frames : dimAt(ref_attr, 2, 18);
}

Eigen::VectorXf AvMossformerRknn::runWithRefFeat(const Eigen::Ref<const Eigen::VectorXf>& mixture,
                                                 const std::vector<float>& ref_feat) {
  const int T = static_cast<int>(mixture.size());
  if (T <= 0) {
    return Eigen::VectorXf(0);
  }

  std::vector<float> mix_flat(static_cast<size_t>(audio_len_), 0.f);
  const int copy_t = std::min(T, audio_len_);
  for (int t = 0; t < copy_t; ++t) {
    mix_flat[static_cast<size_t>(t)] = mixture[t];
  }

  const size_t feat_elems =
      static_cast<size_t>(ref_feat_channels_) * static_cast<size_t>(ref_frames_);
  std::vector<float> feat_flat(feat_elems, 0.f);
  const size_t copy_n = std::min(ref_feat.size(), feat_elems);
  if (copy_n > 0) {
    std::memcpy(feat_flat.data(), ref_feat.data(), copy_n * sizeof(float));
  }

  rknn_input inputs[2]{};
  inputs[0].index = 0;
  inputs[0].type = session_->input_attrs()[0].type;
  inputs[0].fmt = session_->input_attrs()[0].fmt;
  inputs[0].size = static_cast<uint32_t>(mix_flat.size() * sizeof(float));
  inputs[0].buf = mix_flat.data();
  inputs[0].pass_through = 0;

  inputs[1].index = 1;
  inputs[1].type = session_->input_attrs()[1].type;
  inputs[1].fmt = session_->input_attrs()[1].fmt;
  inputs[1].size = static_cast<uint32_t>(feat_flat.size() * sizeof(float));
  inputs[1].buf = feat_flat.data();
  inputs[1].pass_through = 0;

  std::vector<rknn_output> outputs(session_->io_num().n_output);
  for (uint32_t i = 0; i < session_->io_num().n_output; ++i) {
    outputs[i].index = i;
    outputs[i].want_float = 1;
    outputs[i].is_prealloc = 0;
  }

  session_->run({inputs[0], inputs[1]}, outputs);

  const float* out_ptr = reinterpret_cast<const float*>(outputs[0].buf);
  const rknn_tensor_attr& out_attr = session_->output_attrs()[0];
  int out_t = copy_t;
  if (out_attr.n_dims >= 2) {
    const int last = dimAt(out_attr, out_attr.n_dims - 1, copy_t);
    if (last > 0) {
      out_t = std::min(copy_t, last);
    }
  }
  out_t = std::min(out_t, T);

  Eigen::VectorXf y(out_t);
  for (int i = 0; i < out_t; ++i) {
    y[i] = out_ptr[i];
  }

  for (auto& o : outputs) {
    if (o.is_prealloc == 0 && o.buf != nullptr) {
      free(o.buf);
    }
  }
  return y;
}

Eigen::VectorXf AvMossformerRknn::run(const Eigen::Ref<const Eigen::VectorXf>& mixture,
                                      const std::vector<cv::Mat>& ref_frames) {
  (void)ref_frames;
  throw std::runtime_error(
      "AvMossformerRknn::run(frames) requires AvMossformerSplit with ref ONNX + RKNN sep");
}

}  // namespace av_tse
