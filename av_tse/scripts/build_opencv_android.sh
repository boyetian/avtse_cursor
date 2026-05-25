#!/usr/bin/env bash
# Android: omit videoio — OpenCV 4.8 install does not ship libade.a that videoio's
# export graph references; libav_tse sources do not use VideoCapture.
# Linux / FetchContent: keep videoio for cpp_test_av_tse (mp4).
# Output layout matches rknn_model_zoo: .../sdk/native/jni/abi-<ABI>/OpenCVConfig.cmake
#
# Usage:
#   export ANDROID_NDK_PATH=~/other/android-ndk-r19c   # r19 recommended (same as rknn_model_zoo)
#   ./scripts/build_opencv_android.sh
#
# Then configure av_tse with:
#   -DAV_TSE_ANDROID_OPENCV_DIR=$INSTALL_PREFIX/sdk/native/jni/abi-arm64-v8a
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AV_TSE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
THIRD_PARTY="${AV_TSE_DIR}/third_party"

if [[ -z "${ANDROID_NDK_PATH:-}" ]]; then
  if [[ -d "${HOME}/other/android-ndk-r19c" ]]; then
    export ANDROID_NDK_PATH="${HOME}/other/android-ndk-r19c"
  else
    echo "Set ANDROID_NDK_PATH to your Android NDK (r18/r19)." >&2
    exit 1
  fi
fi

TOOLCHAIN="${ANDROID_NDK_PATH}/build/cmake/android.toolchain.cmake"
if [[ ! -f "${TOOLCHAIN}" ]]; then
  echo "Missing NDK CMake toolchain: ${TOOLCHAIN}" >&2
  exit 1
fi

OPENCV_VERSION="${OPENCV_VERSION:-4.8.0}"
ANDROID_ABI="${ANDROID_ABI:-arm64-v8a}"
ANDROID_PLATFORM="${ANDROID_PLATFORM:-android-24}"
# Install prefix (layout: $INSTALL_PREFIX/sdk/native/jni/abi-$ANDROID_ABI/)
INSTALL_PREFIX="${INSTALL_PREFIX:-${THIRD_PARTY}/opencv-${OPENCV_VERSION}-android-${ANDROID_ABI}}"
OPENCV_SRC="${OPENCV_SRC:-${THIRD_PARTY}/opencv-${OPENCV_VERSION}}"
BUILD_DIR="${BUILD_DIR:-${THIRD_PARTY}/opencv-${OPENCV_VERSION}-build-android-${ANDROID_ABI}}"

NPROC="$(nproc 2>/dev/null || echo 4)"

echo "ANDROID_NDK_PATH=${ANDROID_NDK_PATH}"
echo "OPENCV_SRC=${OPENCV_SRC}"
echo "BUILD_DIR=${BUILD_DIR}"
echo "INSTALL_PREFIX=${INSTALL_PREFIX}"
echo "ANDROID_ABI=${ANDROID_ABI} ANDROID_PLATFORM=${ANDROID_PLATFORM}"

mkdir -p "${THIRD_PARTY}"
if [[ ! -f "${OPENCV_SRC}/CMakeLists.txt" ]]; then
  echo "Cloning OpenCV ${OPENCV_VERSION} into ${OPENCV_SRC} ..."
  git clone --depth 1 --branch "${OPENCV_VERSION}" https://github.com/opencv/opencv.git "${OPENCV_SRC}"
fi

rm -rf "${BUILD_DIR}"
rm -rf "${INSTALL_PREFIX}"
cmake -S "${OPENCV_SRC}" -B "${BUILD_DIR}" \
  -DCMAKE_TOOLCHAIN_FILE="${TOOLCHAIN}" \
  -DANDROID_ABI="${ANDROID_ABI}" \
  -DANDROID_PLATFORM="${ANDROID_PLATFORM}" \
  -DANDROID_STL=c++_static \
  -DCMAKE_INSTALL_PREFIX="${INSTALL_PREFIX}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_SHARED_LIBS=OFF \
  -DBUILD_ANDROID_EXAMPLES=OFF \
  -DINSTALL_ANDROID_EXAMPLES=OFF \
  -DBUILD_TESTS=OFF \
  -DBUILD_PERF_TESTS=OFF \
  -DBUILD_opencv_apps=OFF \
  -DBUILD_opencv_java=OFF \
  -DBUILD_JAVA=OFF \
  -DBUILD_opencv_python2=OFF \
  -DOPENCV_GENERATE_PKGCONFIG=OFF \
  -DWITH_CUDA=OFF \
  -DWITH_OPENCL=OFF \
  -DWITH_IPP=OFF \
  -DWITH_ITT=OFF \
  -DWITH_FFMPEG=OFF \
  -DWITH_GTK=OFF \
  -DWITH_QT=OFF \
  -DOPENCV_FORCE_3RDPARTY_BUILD=ON \
  -DWITH_PROTOBUF=OFF \
  -DBUILD_PROTOBUF=OFF \
  -DWITH_ADE=OFF \
  -DBUILD_LIST=core,imgproc,imgcodecs,objdetect,features2d,calib3d

cmake --build "${BUILD_DIR}" -j"${NPROC}"
cmake --install "${BUILD_DIR}"

ABI_DIR="${INSTALL_PREFIX}/sdk/native/jni/abi-${ANDROID_ABI}"
if [[ ! -f "${ABI_DIR}/OpenCVConfig.cmake" ]]; then
  echo "Install layout unexpected; OpenCVConfig.cmake not at ${ABI_DIR}" >&2
  find "${INSTALL_PREFIX}" -name OpenCVConfig.cmake 2>/dev/null | head -5 >&2 || true
  exit 1
fi

echo ""
echo "Done. Point CMake at:"
echo "  -DAV_TSE_ANDROID_OPENCV_DIR=${ABI_DIR}"
echo ""
echo "Example full av_tse (RKNN) for Android:"
echo "  cmake -B build -S ${AV_TSE_DIR} \\"
echo "    -DAV_TSE_INFERENCE_BACKEND=RKNN \\"
echo "    -DAV_TSE_ANDROID_OPENCV_DIR=${ABI_DIR} \\"
echo "    -DAV_TSE_BUILD_LIBRARY=ON \\"
echo "    -DAV_TSE_BUILD_SMOKE=ON \\"
echo "    -DCMAKE_SYSTEM_NAME=Android -DCMAKE_SYSTEM_VERSION=23 \\"
echo "    -DCMAKE_ANDROID_ARCH_ABI=${ANDROID_ABI} \\"
echo "    -DCMAKE_ANDROID_STL_TYPE=c++_static \\"
echo "    -DCMAKE_ANDROID_NDK=\${ANDROID_NDK_PATH} \\"
echo "    -DRKNN_MODEL_ZOO_ROOT=\${RKNN_MODEL_ZOO_ROOT}   # 或 -DRKNN_RKNPU2_ROOT=.../rknpu2"
