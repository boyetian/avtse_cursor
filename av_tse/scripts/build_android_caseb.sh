#!/bin/bash
# Cross-compile av_tse_caseb (Case B: wav + mp4 + StreamInferenceSDK) for RK3588 Android arm64-v8a.
# Package install bundle for: adb push ... /data/av_tse_caseb && ./av_tse_caseb
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AV_TSE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
SPEECH_ROOT="$(cd "${AV_TSE_DIR}/../.." && pwd)"

if [[ -z "${ANDROID_NDK_PATH:-}" ]]; then
  if [[ -d "${HOME}/other/android-ndk-r19c" ]]; then
    export ANDROID_NDK_PATH="${HOME}/other/android-ndk-r19c"
  else
    echo "Set ANDROID_NDK_PATH (e.g. ~/other/android-ndk-r19c)" >&2
    exit 1
  fi
fi

if [[ -z "${RKNN_RKNPU2_ROOT:-}" ]]; then
  if [[ -z "${RKNN_MODEL_ZOO_ROOT:-}" ]]; then
    if [[ -d "${HOME}/workspace/rknn_model_zoo" ]]; then
      export RKNN_MODEL_ZOO_ROOT="${HOME}/workspace/rknn_model_zoo"
    else
      echo "Set RKNN_RKNPU2_ROOT or RKNN_MODEL_ZOO_ROOT" >&2
      exit 1
    fi
  fi
fi

if [[ -z "${AV_TSE_ANDROID_OPENCV_DIR:-}" ]]; then
  _default_ocv="${AV_TSE_DIR}/third_party/opencv-4.8.0-android-arm64-v8a/sdk/native/jni/abi-arm64-v8a"
  if [[ -f "${_default_ocv}/OpenCVConfig.cmake" ]]; then
    export AV_TSE_ANDROID_OPENCV_DIR="${_default_ocv}"
  else
    echo "Set AV_TSE_ANDROID_OPENCV_DIR (build with scripts/build_opencv_android.sh first)" >&2
    exit 1
  fi
fi

# Host assets: Python av_tse tree (测试用例 + checkpoints)
if [[ -z "${AV_TSE_ASSETS_SOURCE:-}" ]]; then
  if [[ -d "${SPEECH_ROOT}/av_tse" ]]; then
    export AV_TSE_ASSETS_SOURCE="${SPEECH_ROOT}/av_tse"
  else
    echo "Set AV_TSE_ASSETS_SOURCE to folder containing 测试用例/ and checkpoints/" >&2
    exit 1
  fi
fi

TARGET_ARCH="${TARGET_ARCH:-arm64-v8a}"
BUILD_TYPE="${BUILD_TYPE:-Release}"
BUILD_DIR="${BUILD_DIR:-${AV_TSE_DIR}/build-android-caseb}"
INSTALL_DIR="${INSTALL_DIR:-${AV_TSE_DIR}/install/rk3588_android_${TARGET_ARCH}/av_tse_caseb}"
DEVICE_ROOT="${DEVICE_ROOT:-/data/av_tse_caseb}"

echo "ANDROID_NDK_PATH=${ANDROID_NDK_PATH}"
echo "AV_TSE_ANDROID_OPENCV_DIR=${AV_TSE_ANDROID_OPENCV_DIR}"
echo "AV_TSE_ASSETS_SOURCE=${AV_TSE_ASSETS_SOURCE}"
echo "BUILD_DIR=${BUILD_DIR}"
echo "INSTALL_DIR=${INSTALL_DIR}"

CMAKE_EXTRA=()
if [[ -n "${RKNN_RKNPU2_ROOT:-}" ]]; then
  CMAKE_EXTRA+=(-DRKNN_RKNPU2_ROOT="${RKNN_RKNPU2_ROOT}")
fi
if [[ -n "${RKNN_MODEL_ZOO_ROOT:-}" ]]; then
  CMAKE_EXTRA+=(-DRKNN_MODEL_ZOO_ROOT="${RKNN_MODEL_ZOO_ROOT}")
fi

cmake -B "${BUILD_DIR}" -S "${AV_TSE_DIR}" \
  -DAV_TSE_INFERENCE_BACKEND=RKNN \
  -DAV_TSE_BUILD_LIBRARY=ON \
  -DAV_TSE_BUILD_CASEB=ON \
  -DAV_TSE_BUILD_SMOKE=OFF \
  -DAV_TSE_ANDROID_OPENCV_DIR="${AV_TSE_ANDROID_OPENCV_DIR}" \
  -DAV_TSE_CASEB_DATA_ROOT="${DEVICE_ROOT}" \
  "${CMAKE_EXTRA[@]}" \
  -DCMAKE_SYSTEM_NAME=Android \
  -DCMAKE_SYSTEM_VERSION=23 \
  -DCMAKE_ANDROID_ARCH_ABI="${TARGET_ARCH}" \
  -DCMAKE_ANDROID_STL_TYPE=c++_static \
  -DCMAKE_ANDROID_NDK="${ANDROID_NDK_PATH}" \
  -DCMAKE_BUILD_TYPE="${BUILD_TYPE}"

cmake --build "${BUILD_DIR}" -j"$(nproc)" --target av_tse_caseb

rm -rf "${INSTALL_DIR}"
mkdir -p "${INSTALL_DIR}/lib" "${INSTALL_DIR}/audio" "${INSTALL_DIR}/video" \
  "${INSTALL_DIR}/model" "${INSTALL_DIR}/out"

cp "${BUILD_DIR}/av_tse_caseb" "${INSTALL_DIR}/"

if [[ -n "${RKNN_RKNPU2_ROOT:-}" ]]; then
  _RKNN_BASE="${RKNN_RKNPU2_ROOT}"
else
  _RKNN_BASE="${RKNN_MODEL_ZOO_ROOT}/3rdparty/rknpu2"
fi
if [[ -f "${_RKNN_BASE}/Android/${TARGET_ARCH}/librknnrt.so" ]]; then
  _RKNN_RT="${_RKNN_BASE}/Android/${TARGET_ARCH}/librknnrt.so"
elif [[ -f "${_RKNN_BASE}/runtime/Android/librknn_api/${TARGET_ARCH}/librknnrt.so" ]]; then
  _RKNN_RT="${_RKNN_BASE}/runtime/Android/librknn_api/${TARGET_ARCH}/librknnrt.so"
else
  echo "librknnrt.so not found under ${_RKNN_BASE}" >&2
  exit 1
fi
cp "${_RKNN_RT}" "${INSTALL_DIR}/lib/"

_src_audio="${AV_TSE_ASSETS_SOURCE}/测试用例/音频/test03.wav"
_src_video="${AV_TSE_ASSETS_SOURCE}/测试用例/视频/test03.mp4"
_src_cfg="${AV_TSE_ASSETS_SOURCE}/checkpoints/AV_Mossformer/config.yaml"
_src_rknn="${AV_TSE_ASSETS_SOURCE}/checkpoints/AV_Mossformer/av_mossformer2.rknn"

for f in "${_src_audio}" "${_src_video}" "${_src_cfg}"; do
  if [[ ! -f "${f}" ]]; then
    echo "Missing asset: ${f}" >&2
    exit 1
  fi
done
if [[ ! -f "${_src_rknn}" ]]; then
  echo "Missing RKNN model: ${_src_rknn}" >&2
  echo "Convert with scripts/convert_av_mossformer_rknn.py first." >&2
  exit 1
fi

_ffmpeg=""
for c in ffmpeg "${FFMPEG_PATH:-}" "${HOME}/workspace/FFmpeg/build/ffmpeg"; do
  [[ -n "${c}" ]] && command -v "${c}" >/dev/null 2>&1 && _ffmpeg="${c}" && break
  [[ -n "${c}" && -x "${c}" ]] && _ffmpeg="${c}" && break
done
if [[ -z "${_ffmpeg}" ]]; then
  echo "Host ffmpeg required to pre-extract frames (set FFMPEG_PATH or install ffmpeg)." >&2
  exit 1
fi

_frames="${INSTALL_DIR}/video/test03_frames"
mkdir -p "${_frames}"
echo "Pre-extracting video frames (host ffmpeg) -> ${_frames}"
"${_ffmpeg}" -loglevel error -y -i "${_src_video}" "${_frames}/frame_%06d.jpg"
_n_frames=$(find "${_frames}" -maxdepth 1 -name '*.jpg' | wc -l)
if [[ "${_n_frames}" -lt 1 ]]; then
  echo "ffmpeg produced no frames under ${_frames}" >&2
  exit 1
fi
echo "Pre-extracted ${_n_frames} frames."

cp "${_src_audio}" "${INSTALL_DIR}/audio/"
cp "${_src_cfg}" "${INSTALL_DIR}/model/"
cp "${_src_rknn}" "${INSTALL_DIR}/model/"

cat > "${INSTALL_DIR}/README.txt" <<EOF
Push entire directory to device (e.g. ${DEVICE_ROOT}):

  adb root && adb remount
  adb push ${INSTALL_DIR} ${DEVICE_ROOT}

Run on board (video = pre-extracted JPGs only; no mp4/ffmpeg on device):

  adb shell
  cd ${DEVICE_ROOT}
  export LD_LIBRARY_PATH=./lib
  ./av_tse_caseb

Requires: video/test03_frames/frame_*.jpg (created on host during packaging).

Output: ${DEVICE_ROOT}/out/test03_out.wav

Optional env: AV_TSE_CASEB_FPS=30  AV_TSE_CASEB_MODEL_PATH=...
EOF

echo ""
echo "Build complete. Install bundle: ${INSTALL_DIR}"
echo "  adb push ${INSTALL_DIR} ${DEVICE_ROOT}"
echo "  adb shell 'cd ${DEVICE_ROOT} && export LD_LIBRARY_PATH=./lib && ./av_tse_caseb'"
