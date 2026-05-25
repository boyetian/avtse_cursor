#pragma once

#if defined(AV_TSE_USE_RKNN) && AV_TSE_USE_RKNN
#include "av_tse/av_mossformer_rknn.hpp"
#else
#include "av_tse/av_mossformer_onnx.hpp"
#endif

#include <Eigen/Dense>
#include <opencv2/core.hpp>

#include <vector>

namespace av_tse {

#if defined(AV_TSE_USE_RKNN) && AV_TSE_USE_RKNN
using AvMossformerModel = AvMossformerRknn;
#else
using AvMossformerModel = AvMossformerOnnx;
#endif

template <typename Model>
void warmupIngressForward(Model& model, int image_size, int audio_sr, float ref_sr, int hop_samples,
                          int context_samples, int lookahead_samples);

}  // namespace av_tse

#include "av_tse/av_mossformer_warmup.inl"
