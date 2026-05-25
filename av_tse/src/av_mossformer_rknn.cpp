#include "av_tse/av_mossformer_rknn.hpp"

#include "asr_frontend/rknn_session.hpp"

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
  ref_frames_ = ref_frames > 0 ? ref_frames : dimAt(ref_attr, 1, 20);
  image_size_ = dimAt(ref_attr, 2, image_size);
  const int ref_h = dimAt(ref_attr, 3, image_size_);
  const int ref_w = dimAt(ref_attr, 4, image_size_);
  channels_ = dimAt(ref_attr, 5, 3);
  if (ref_h != image_size_ || ref_w != image_size_) {
    throw std::runtime_error("AvMossformerRknn: ref H/W must match image_size from RKNN model");
  }
}

Eigen::VectorXf AvMossformerRknn::run(const Eigen::Ref<const Eigen::VectorXf>& mixture,
                                      const std::vector<cv::Mat>& ref_frames) {
  const int T = static_cast<int>(mixture.size());
  const int Tv = static_cast<int>(ref_frames.size());
  if (T <= 0 || Tv <= 0) {
    return Eigen::VectorXf(0);
  }

  const int H = image_size_;
  const int W = image_size_;
  const int C = channels_;

  std::vector<float> mix_flat(static_cast<size_t>(audio_len_), 0.f);
  const int copy_t = std::min(T, audio_len_);
  for (int t = 0; t < copy_t; ++t) {
    mix_flat[static_cast<size_t>(t)] = mixture[t];
  }

  std::vector<float> ref_flat(static_cast<size_t>(ref_frames_) * static_cast<size_t>(H) *
                              static_cast<size_t>(W) * static_cast<size_t>(C),
                            0.f);
  const int copy_tv = std::min(Tv, ref_frames_);
  for (int t = 0; t < copy_tv; ++t) {
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
  inputs[1].size = static_cast<uint32_t>(ref_flat.size() * sizeof(float));
  inputs[1].buf = ref_flat.data();
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

  rknn_outputs_release(session_->ctx(), static_cast<uint32_t>(outputs.size()), outputs.data());

  if (out_t < T) {
    Eigen::VectorXf padded(T);
    padded.setZero();
    padded.head(out_t) = y;
    return padded;
  }
  return y;
}

}  // namespace av_tse
