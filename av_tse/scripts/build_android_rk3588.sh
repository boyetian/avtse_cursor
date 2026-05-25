#!/bin/bash
# Cross-compile av_tse for RK3588 Android (arm64-v8a).
#
# Backend (default RKNN):
#   export INFERENCE_BACKEND=RKNN   # needs RKNN_MODEL_ZOO_ROOT or RKNN_RKNPU2_ROOT
#   export INFERENCE_BACKEND=ONNX   # CPU debug; smoke not built (use cpp/test board tests)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AV_TSE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [[ -z "${ANDROID_NDK_PATH:-}" ]]; then
  if [[ -d "${HOME}/other/android-ndk-r19c" ]]; then
    export ANDROID_NDK_PATH="${HOME}/other/android-ndk-r19c"
  else
    echo "Set ANDROID_NDK_PATH (e.g. ~/other/android-ndk-r19c)" >&2
    exit 1
  fi
fi

INFERENCE_BACKEND="${INFERENCE_BACKEND:-RKNN}"

# RKNN SDK: either RKNN_RKNPU2_ROOT or RKNN_MODEL_ZOO_ROOT (only when INFERENCE_BACKEND=RKNN).
if [[ "${INFERENCE_BACKEND}" == "RKNN" ]]; then
  if [[ -z "${RKNN_RKNPU2_ROOT:-}" ]]; then
    if [[ -z "${RKNN_MODEL_ZOO_ROOT:-}" ]]; then
      if [[ -d "${HOME}/workspace/rknn_model_zoo" ]]; then
        export RKNN_MODEL_ZOO_ROOT="${HOME}/workspace/rknn_model_zoo"
      else
        echo "Set RKNN_RKNPU2_ROOT (Rockchip rknpu2 SDK) or RKNN_MODEL_ZOO_ROOT (rknn_model_zoo checkout)." >&2
        exit 1
      fi
    fi
  fi
fi

if [[ "${AV_TSE_BUILD_FULL:-}" == "1" && -z "${AV_TSE_ANDROID_OPENCV_DIR:-}" ]]; then
  echo "AV_TSE_BUILD_FULL=1 requires AV_TSE_ANDROID_OPENCV_DIR (self-built OpenCV with objdetect)." >&2
  exit 1
fi
TARGET_ARCH="${TARGET_ARCH:-arm64-v8a}"
BUILD_TYPE="${BUILD_TYPE:-Release}"
BUILD_DIR="${BUILD_DIR:-${AV_TSE_DIR}/build-android-${INFERENCE_BACKEND,,}}"
INSTALL_DIR="${INSTALL_DIR:-${AV_TSE_DIR}/install/rk3588_android_${TARGET_ARCH}/av_tse_smoke}"

ANDROID_TOOLCHAIN="${ANDROID_NDK_PATH}/build/cmake/android.toolchain.cmake"
if [[ ! -f "${ANDROID_TOOLCHAIN}" ]]; then
  echo "Missing NDK toolchain: ${ANDROID_TOOLCHAIN}" >&2
  exit 1
fi

if [[ "${INFERENCE_BACKEND}" == "ONNX" ]]; then
  ANDROID_PLATFORM=android-28
  ANDROID_STL=c++_shared
else
  ANDROID_PLATFORM=android-23
  ANDROID_STL=c++_static
fi

echo "ANDROID_NDK_PATH=${ANDROID_NDK_PATH}"
echo "INFERENCE_BACKEND=${INFERENCE_BACKEND}"
if [[ -n "${RKNN_RKNPU2_ROOT:-}" ]]; then
  echo "RKNN_RKNPU2_ROOT=${RKNN_RKNPU2_ROOT}"
fi
if [[ -n "${RKNN_MODEL_ZOO_ROOT:-}" ]]; then
  echo "RKNN_MODEL_ZOO_ROOT=${RKNN_MODEL_ZOO_ROOT}"
fi
echo "BUILD_DIR=${BUILD_DIR}"
echo "INSTALL_DIR=${INSTALL_DIR}"
if [[ -n "${AV_TSE_ANDROID_OPENCV_DIR:-}" ]]; then
  echo "AV_TSE_ANDROID_OPENCV_DIR=${AV_TSE_ANDROID_OPENCV_DIR}"
fi

CMAKE_EXTRA=()
if [[ "${INFERENCE_BACKEND}" == "ONNX" ]]; then
  CMAKE_EXTRA+=(-DAV_TSE_BUILD_SMOKE=OFF)
elif [[ -n "${RKNN_RKNPU2_ROOT:-}" ]]; then
  CMAKE_EXTRA+=(-DRKNN_RKNPU2_ROOT="${RKNN_RKNPU2_ROOT}")
fi
if [[ -n "${RKNN_MODEL_ZOO_ROOT:-}" ]]; then
  CMAKE_EXTRA+=(-DRKNN_MODEL_ZOO_ROOT="${RKNN_MODEL_ZOO_ROOT}")
fi
if [[ -n "${AV_TSE_ANDROID_OPENCV_DIR:-}" ]]; then
  CMAKE_EXTRA+=(-DAV_TSE_ANDROID_OPENCV_DIR="${AV_TSE_ANDROID_OPENCV_DIR}")
fi
# Optional: AV_TSE_BUILD_FULL=1 with AV_TSE_ANDROID_OPENCV_DIR to build libav_tse + smoke.
if [[ "${INFERENCE_BACKEND}" != "ONNX" ]]; then
  if [[ "${AV_TSE_BUILD_FULL:-}" == "1" ]]; then
    CMAKE_EXTRA+=(-DAV_TSE_BUILD_LIBRARY=ON -DAV_TSE_BUILD_SMOKE=ON)
  else
    CMAKE_EXTRA+=(-DAV_TSE_BUILD_LIBRARY=OFF -DAV_TSE_BUILD_SMOKE=ON)
  fi
fi

cmake -B "${BUILD_DIR}" -S "${AV_TSE_DIR}" \
  -DCMAKE_TOOLCHAIN_FILE="${ANDROID_TOOLCHAIN}" \
  -DANDROID_ABI="${TARGET_ARCH}" \
  -DANDROID_PLATFORM="${ANDROID_PLATFORM}" \
  -DANDROID_STL="${ANDROID_STL}" \
  -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" \
  -DAV_TSE_INFERENCE_BACKEND="${INFERENCE_BACKEND}" \
  "${CMAKE_EXTRA[@]}"

if [[ "${INFERENCE_BACKEND}" == "ONNX" ]]; then
  cmake --build "${BUILD_DIR}" -j"$(nproc)"
  echo "ONNX backend: use cpp/test/scripts/build_android_onnx_test.sh for board GTests."
  exit 0
fi

if [[ "${AV_TSE_BUILD_FULL:-}" == "1" ]]; then
  cmake --build "${BUILD_DIR}" -j"$(nproc)"
else
  cmake --build "${BUILD_DIR}" -j"$(nproc)" --target av_tse_smoke
fi

rm -rf "${INSTALL_DIR}"
mkdir -p "${INSTALL_DIR}/lib" "${INSTALL_DIR}/model"
cp "${BUILD_DIR}/av_tse_smoke" "${INSTALL_DIR}/"
if [[ "${AV_TSE_BUILD_FULL:-}" == "1" && -f "${BUILD_DIR}/libav_tse.a" ]]; then
  cp "${BUILD_DIR}/libav_tse.a" "${INSTALL_DIR}/lib/"
fi
if [[ -n "${RKNN_RKNPU2_ROOT:-}" ]]; then
  _RKNN_BASE="${RKNN_RKNPU2_ROOT}"
elif [[ -n "${RKNN_MODEL_ZOO_ROOT:-}" ]]; then
  _RKNN_BASE="${RKNN_MODEL_ZOO_ROOT}/3rdparty/rknpu2"
fi
if [[ -f "${_RKNN_BASE}/Android/${TARGET_ARCH}/librknnrt.so" ]]; then
  _RKNN_RT="${_RKNN_BASE}/Android/${TARGET_ARCH}/librknnrt.so"
elif [[ -f "${_RKNN_BASE}/runtime/Android/librknn_api/${TARGET_ARCH}/librknnrt.so" ]]; then
  _RKNN_RT="${_RKNN_BASE}/runtime/Android/librknn_api/${TARGET_ARCH}/librknnrt.so"
else
  echo "librknnrt.so not found under ${_RKNN_BASE} (tried rknn_model_zoo and rknn-toolkit2 runtime layouts)." >&2
  exit 1
fi
cp "${_RKNN_RT}" "${INSTALL_DIR}/lib/"
cp "${AV_TSE_DIR}/scripts/README_MODELS.txt" "${INSTALL_DIR}/model/"

echo ""
echo "Build complete. Push to device:"
echo "  adb root && adb remount"
echo "  adb push ${INSTALL_DIR} /data/"
echo "  adb shell 'cd /data/av_tse_smoke && export LD_LIBRARY_PATH=./lib && ./av_tse_smoke model/av_mossformer2.rknn'"
