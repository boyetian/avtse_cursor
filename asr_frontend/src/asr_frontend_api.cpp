#include "asr_frontend/asr_frontend_api.hpp"

#ifndef ASR_FRONTEND_USE_RKNN
#include "asr_frontend/onnx_runtime_setup.hpp"
#endif

#include <stdexcept>

namespace asr_frontend {

AsrFrontendApi::AsrFrontendApi(bool enable_aec, bool enable_speech_enhancement,
                               const AsrFrontendPaths& paths, float se_decode_window_sec,
                               int target_sr, int ort_intra_aec, int ort_intra_se)
    : target_sr_(target_sr), se_decode_window_sec_(se_decode_window_sec) {
#if defined(ASR_FRONTEND_USE_RKNN) && ASR_FRONTEND_USE_RKNN
  (void)ort_intra_aec;
  (void)ort_intra_se;
  if (enable_aec) {
    aec_ = std::make_unique<DfsmnAecRknn>(paths.dfsmn_aec_rknn);
  }
  if (enable_speech_enhancement) {
    se_ = std::make_unique<ZipEnhancerFullRknn>(paths.zipenhancer_full_rknn,
                                                se_decode_window_sec, target_sr);
  }
#else
  detail::relax_onnx_released_opset_check_if_unset();
  if (enable_aec) {
    aec_ = std::make_unique<DfsmnAecOnnx>(paths.dfsmn_aec_onnx, ort_intra_aec);
  }
  if (enable_speech_enhancement) {
    se_ = std::make_unique<ZipEnhancerFullOnnx>(paths.zipenhancer_full_onnx,
                                               se_decode_window_sec, target_sr,
                                               ort_intra_se);
  }
#endif
}

Eigen::VectorXf AsrFrontendApi::call_aec_stream(const Eigen::Ref<const Eigen::MatrixXf>& nearend,
                                                const Eigen::Ref<const Eigen::MatrixXf>& farend) {
  if (!aec_) {
    return nearend.colwise().mean().transpose();
  }
  auto i16 = aec_->decode_stream(nearend, farend);
  Eigen::VectorXf f(i16.size());
  for (Eigen::Index i = 0; i < i16.size(); ++i) {
    f[i] = static_cast<float>(i16[i]) / 32768.0f;
  }
  return f;
}

std::vector<Eigen::VectorXf> AsrFrontendApi::call_se_stream(const float* wav_bct, int B, int C,
                                                             int T, bool is_first, bool is_last) {
  if (!se_) {
    throw std::runtime_error("AsrFrontendApi::call_se_stream: SE disabled");
  }
  return se_->decode_stream(wav_bct, B, C, T, is_first, is_last);
}

}  // namespace asr_frontend
