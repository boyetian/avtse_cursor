#!/bin/bash
# Cross-compile asr_frontend for RK3588 Android.
#
# Backend (default RKNN, NPU):
#   export INFERENCE_BACKEND=RKNN   # needs RKNN_MODEL_ZOO_ROOT
#   export INFERENCE_BACKEND=ONNX   # CPU debug; CMake downloads onnxruntime-android AAR
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ASR_FRONTEND_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [[ -z "${ANDROID_NDK_PATH:-}" ]]; then
  if [[ -d "${HOME}/other/android-ndk-r19c" ]]; then
    export ANDROID_NDK_PATH="${HOME}/other/android-ndk-r19c"
  else
    echo "Set ANDROID_NDK_PATH (e.g. ~/other/android-ndk-r19c)" >&2
    exit 1
  fi
fi

INFERENCE_BACKEND="${INFERENCE_BACKEND:-RKNN}"

if [[ "${INFERENCE_BACKEND}" == "RKNN" ]]; then
  if [[ -z "${RKNN_MODEL_ZOO_ROOT:-}" ]]; then
    if [[ -d "${HOME}/workspace/rknn_model_zoo" ]]; then
      export RKNN_MODEL_ZOO_ROOT="${HOME}/workspace/rknn_model_zoo"
    else
      echo "Set RKNN_MODEL_ZOO_ROOT to rknn_model_zoo checkout" >&2
      exit 1
    fi
  fi
fi

TARGET_ARCH="${TARGET_ARCH:-arm64-v8a}"
BUILD_TYPE="${BUILD_TYPE:-Release}"
BUILD_DIR="${BUILD_DIR:-${ASR_FRONTEND_DIR}/build-android-${INFERENCE_BACKEND,,}}"
INSTALL_DIR="${INSTALL_DIR:-${ASR_FRONTEND_DIR}/install/rk3588_android_${TARGET_ARCH}/asr_frontend_smoke}"

ANDROID_TOOLCHAIN="${ANDROID_NDK_PATH}/build/cmake/android.toolchain.cmake"
if [[ ! -f "${ANDROID_TOOLCHAIN}" ]]; then
  echo "Missing NDK toolchain: ${ANDROID_TOOLCHAIN}" >&2
  exit 1
fi

CMAKE_EXTRA=()
if [[ "${INFERENCE_BACKEND}" == "RKNN" ]]; then
  CMAKE_EXTRA+=(-DTARGET_SOC=rk3588 -DRKNN_MODEL_ZOO_ROOT="${RKNN_MODEL_ZOO_ROOT}")
  ANDROID_PLATFORM=android-23
  ANDROID_STL=c++_static
else
  ANDROID_PLATFORM=android-28
  ANDROID_STL=c++_shared
fi

_ORT_ROOT="${BUILD_DIR}/onnxruntime-android-1.17.1"
CMAKE_ORT=()
if [[ "${INFERENCE_BACKEND}" == "ONNX" && -d "${_ORT_ROOT}/headers" ]]; then
  CMAKE_ORT+=(-DONNXRUNTIME_ROOT="${_ORT_ROOT}")
fi

echo "ANDROID_NDK_PATH=${ANDROID_NDK_PATH}"
echo "INFERENCE_BACKEND=${INFERENCE_BACKEND}"
if [[ "${INFERENCE_BACKEND}" == "RKNN" ]]; then
  echo "RKNN_MODEL_ZOO_ROOT=${RKNN_MODEL_ZOO_ROOT}"
fi
echo "BUILD_DIR=${BUILD_DIR}"
echo "INSTALL_DIR=${INSTALL_DIR}"

cmake -B "${BUILD_DIR}" -S "${ASR_FRONTEND_DIR}" \
  -DCMAKE_TOOLCHAIN_FILE="${ANDROID_TOOLCHAIN}" \
  -DANDROID_ABI="${TARGET_ARCH}" \
  -DANDROID_PLATFORM="${ANDROID_PLATFORM}" \
  -DANDROID_STL="${ANDROID_STL}" \
  -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" \
  -DASR_FRONTEND_INFERENCE_BACKEND="${INFERENCE_BACKEND}" \
  -DASR_FRONTEND_BUILD_TESTS=OFF \
  "${CMAKE_ORT[@]}" \
  "${CMAKE_EXTRA[@]}"

cmake --build "${BUILD_DIR}" -j"$(nproc)" --target asr_frontend_smoke

rm -rf "${INSTALL_DIR}"
mkdir -p "${INSTALL_DIR}/lib" "${INSTALL_DIR}/model"
cp "${BUILD_DIR}/asr_frontend_smoke" "${INSTALL_DIR}/"

if [[ "${INFERENCE_BACKEND}" == "RKNN" ]]; then
  _RKNN_RT="${RKNN_MODEL_ZOO_ROOT}/3rdparty/rknpu2/Android/${TARGET_ARCH}/librknnrt.so"
  cp "${_RKNN_RT}" "${INSTALL_DIR}/lib/"
else
  _ORT_SO="${BUILD_DIR}/onnxruntime-android-1.17.1/jni/${TARGET_ARCH}/libonnxruntime.so"
  if [[ -f "${_ORT_SO}" ]]; then
    cp "${_ORT_SO}" "${INSTALL_DIR}/lib/"
  fi
  _NDK_LIBCXX="${ANDROID_NDK_PATH}/sources/cxx-stl/llvm-libc++/libs/${TARGET_ARCH}/libc++_shared.so"
  if [[ -f "${_NDK_LIBCXX}" ]]; then
    cp "${_NDK_LIBCXX}" "${INSTALL_DIR}/lib/"
  fi
fi

cp "${ASR_FRONTEND_DIR}/scripts/README_MODELS.txt" "${INSTALL_DIR}/model/" 2>/dev/null || true

echo ""
echo "Build complete. Push to device:"
echo "  adb root && adb remount"
echo "  adb push ${INSTALL_DIR} /data/"
if [[ "${INFERENCE_BACKEND}" == "RKNN" ]]; then
  echo "  adb shell 'cd /data/asr_frontend_smoke && export LD_LIBRARY_PATH=./lib && ./asr_frontend_smoke model/DFSMN_AEC_opt.rknn model/zipenhancer_full.rknn'"
else
  echo "  adb shell 'cd /data/asr_frontend_smoke && export LD_LIBRARY_PATH=./lib && ./asr_frontend_smoke model/DFSMN_AEC_opt.onnx model/zipenhancer_full.onnx'"
fi
