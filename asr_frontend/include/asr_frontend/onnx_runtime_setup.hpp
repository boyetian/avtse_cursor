#pragma once

#include <cstdlib>
#include <string>

#if defined(_WIN32) && defined(_MSC_VER)
#include <stdlib.h>
#endif

namespace asr_frontend {
namespace detail {

/// If unset, set ONNX Runtime env so models with newer experimental opsets
/// (e.g. `ai.onnx.ml` opset 5 on DFSMN exports) load with a warning instead of
/// `ORT_THROW` (see `model_load_utils::kAllowReleasedONNXOpsetOnly`).
inline void relax_onnx_released_opset_check_if_unset() {
  static const char kVar[] = "ALLOW_RELEASED_ONNX_OPSET_ONLY";
  if (std::getenv(kVar) != nullptr) {
    return;
  }
#if defined(_WIN32) && defined(_MSC_VER)
  _putenv_s(kVar, "0");
#elif defined(_WIN32)
  {
    std::string assign = std::string(kVar) + "=0";
    _putenv(assign.c_str());
  }
#else
  setenv(kVar, "0", 0);
#endif
}

}  // namespace detail
}  // namespace asr_frontend
