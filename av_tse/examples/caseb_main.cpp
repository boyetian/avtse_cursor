#include "av_tse/caseb_runner.hpp"

#include <cstdlib>
#include <iostream>
#include <string>

#ifndef AV_TSE_CASEB_DATA_ROOT
#define AV_TSE_CASEB_DATA_ROOT "/data/av_tse_caseb"
#endif

int main(int argc, char** argv) {
  std::cout << "av_tse_caseb: Case B stream demo (wav + video + StreamInferenceSDK)\n";
#if defined(AV_TSE_USE_RKNN) && AV_TSE_USE_RKNN
  std::cout << "backend: RKNN\n";
#else
  std::cout << "backend: ONNX\n";
#endif

  std::string data_root(AV_TSE_CASEB_DATA_ROOT);
  if (argc >= 2 && argv[1][0] != '\0') {
    data_root = argv[1];
  } else if (const char* env = std::getenv("AV_TSE_CASEB_DATA_ROOT")) {
    if (env[0] != '\0') {
      data_root = env;
    }
  }

  const av_tse::CaseBPaths paths =
      av_tse::resolveCaseBPaths(data_root, av_tse::CaseBLayout::AndroidFlat);
  const int rc = av_tse::runCaseB(paths);
  if (rc != 0) {
    std::cerr << "av_tse_caseb failed with code " << rc << "\n";
  }
  return rc;
}
