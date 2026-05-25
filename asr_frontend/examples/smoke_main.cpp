#include <cstdlib>
#include <iostream>

#include "asr_frontend/stream_inference.hpp"

int main(int argc, char** argv) {
#if defined(ASR_FRONTEND_USE_RKNN) && ASR_FRONTEND_USE_RKNN
  std::cout << "backend: RKNN\n";
#else
  const char* onnx_root = std::getenv("ONNXRUNTIME_ROOT");
  if (!onnx_root) {
    std::cerr << "backend: ONNX (set ONNXRUNTIME_ROOT if loading models fails)\n";
  } else {
    std::cout << "backend: ONNX\n";
  }
#endif

  asr_frontend::StreamInferenceOptions opt;
  if (argc >= 3) {
#if defined(ASR_FRONTEND_USE_RKNN) && ASR_FRONTEND_USE_RKNN
    opt.paths.dfsmn_aec_rknn = argv[1];
    opt.paths.zipenhancer_full_rknn = argv[2];
    std::cout << "aec_rknn: " << opt.paths.dfsmn_aec_rknn << "\n";
    std::cout << "se_rknn:  " << opt.paths.zipenhancer_full_rknn << "\n";
#else
    opt.paths.dfsmn_aec_onnx = argv[1];
    opt.paths.zipenhancer_full_onnx = argv[2];
    std::cout << "aec_onnx: " << opt.paths.dfsmn_aec_onnx << "\n";
    std::cout << "se_onnx:  " << opt.paths.zipenhancer_full_onnx << "\n";
#endif
  }

  try {
    asr_frontend::StreamInference si(opt);
    Eigen::MatrixXf chunk = Eigen::MatrixXf::Zero(1, 1600);
    auto out = si.stream_inference(chunk, std::nullopt, true, false, false, 16000);
    std::cout << "smoke outputs: " << out.size() << "\n";
  } catch (const std::exception& e) {
    std::cerr << "smoke failed: " << e.what() << "\n";
    return 1;
  }
  return 0;
}
