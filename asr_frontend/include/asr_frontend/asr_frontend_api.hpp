#pragma once

#include <Eigen/Dense>
#include <memory>
#include <string>
#include <vector>

#if defined(ASR_FRONTEND_USE_RKNN) && ASR_FRONTEND_USE_RKNN
#include "asr_frontend/dfsmn_aec_rknn.hpp"
#include "asr_frontend/zipenhancer_full_rknn.hpp"
#else
#include "asr_frontend/dfsmn_aec_onnx.hpp"
#include "asr_frontend/zipenhancer_full_onnx.hpp"
#endif

namespace asr_frontend {

struct AsrFrontendPaths {
  std::string dfsmn_aec_onnx = "checkpoints/dfsmn_aec_16k/DFSMN_AEC_opt.onnx";
  std::string zipenhancer_full_onnx =
      "checkpoints/zip_enhancer_se_16k/zipenhancer_full.onnx";
  std::string dfsmn_aec_rknn = "checkpoints/dfsmn_aec_16k/DFSMN_AEC_opt.rknn";
  std::string zipenhancer_full_rknn =
      "checkpoints/zip_enhancer_se_16k/zipenhancer_full.rknn";
};

/// Holds inference sessions for AEC + ZipEnhancer (full), mirroring Python
/// `ASRFrontendAPI` model wiring for the streaming pipeline.
class AsrFrontendApi {
 public:
  AsrFrontendApi(bool enable_aec, bool enable_speech_enhancement,
                 const AsrFrontendPaths& paths = AsrFrontendPaths{},
                 float se_decode_window_sec = 1.0f, int target_sr = 16000,
                 int ort_intra_aec = 8, int ort_intra_se = 16);

  bool enable_aec() const { return static_cast<bool>(aec_); }
  bool enable_se() const { return static_cast<bool>(se_); }
  int target_sr() const { return target_sr_; }
  float se_decode_window_sec() const { return se_decode_window_sec_; }

  /// Returns mono float in [-1,1], length = nearend.cols().
  Eigen::VectorXf call_aec_stream(const Eigen::Ref<const Eigen::MatrixXf>& nearend,
                                  const Eigen::Ref<const Eigen::MatrixXf>& farend);

  /// `wav` layout `[B][C][T]` row-major float. One output vector per batch row.
  std::vector<Eigen::VectorXf> call_se_stream(const float* wav_bct, int B, int C, int T,
                                              bool is_first, bool is_last);

 private:
  int target_sr_{16000};
  float se_decode_window_sec_{1.0f};
#if defined(ASR_FRONTEND_USE_RKNN) && ASR_FRONTEND_USE_RKNN
  std::unique_ptr<DfsmnAecRknn> aec_;
  std::unique_ptr<ZipEnhancerFullRknn> se_;
#else
  std::unique_ptr<DfsmnAecOnnx> aec_;
  std::unique_ptr<ZipEnhancerFullOnnx> se_;
#endif
};

}  // namespace asr_frontend
