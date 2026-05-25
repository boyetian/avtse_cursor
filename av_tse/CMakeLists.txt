cmake_minimum_required(VERSION 3.16)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(AV_TSE_INFERENCE_BACKEND "ONNX" CACHE STRING "Inference backend: ONNX or RKNN")
set_property(CACHE AV_TSE_INFERENCE_BACKEND PROPERTY STRINGS ONNX RKNN)

if(NOT AV_TSE_INFERENCE_BACKEND MATCHES "^(ONNX|RKNN)$")
  message(FATAL_ERROR "AV_TSE_INFERENCE_BACKEND must be ONNX or RKNN")
endif()

option(AV_TSE_BUILD_SMOKE "Build av_tse_smoke (RKNN board test)" ON)
option(AV_TSE_BUILD_CASEB "Build av_tse_caseb (Case B wav+mp4 stream demo for adb)" OFF)
option(AV_TSE_BUILD_LIBRARY "Build full libav_tse static library" ON)

set(AV_TSE_CASEB_DATA_ROOT "/data/av_tse_caseb" CACHE STRING
    "Default on-device data root for av_tse_caseb (audio/, video/, model/, out/)")

set(AV_TSE_ANDROID_OPENCV_DIR "" CACHE PATH
    "OpenCV Android SDK: path to abi folder (e.g. .../sdk/native/jni/abi-arm64-v8a). Enables objdetect for full libav_tse.")

if(CMAKE_SYSTEM_NAME STREQUAL "Android" AND NOT AV_TSE_ANDROID_OPENCV_DIR)
  set(AV_TSE_BUILD_LIBRARY OFF CACHE BOOL
      "Full av_tse on Android needs OpenCV objdetect; build custom OpenCV and set AV_TSE_ANDROID_OPENCV_DIR, or use smoke only"
      FORCE)
endif()

include(FetchContent)

find_package(Eigen3 QUIET CONFIG)
if(NOT Eigen3_FOUND)
  FetchContent_Declare(
    eigen
    GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
    GIT_TAG 3.4.0)
  set(EIGEN_BUILD_DOC OFF CACHE BOOL "" FORCE)
  set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(eigen)
endif()

option(AV_TSE_FETCH_OPENCV "Fetch and build minimal OpenCV when system OpenCV is missing" ON)

if(CMAKE_SYSTEM_NAME STREQUAL "Android")
  if(NOT DEFINED RKNN_MODEL_ZOO_ROOT AND DEFINED ENV{RKNN_MODEL_ZOO_ROOT})
    set(RKNN_MODEL_ZOO_ROOT "$ENV{RKNN_MODEL_ZOO_ROOT}")
  endif()
  if(AV_TSE_ANDROID_OPENCV_DIR)
    set(OpenCV_DIR "${AV_TSE_ANDROID_OPENCV_DIR}" CACHE PATH "OpenCV for Android (abi dir)" FORCE)
    set(AV_TSE_FETCH_OPENCV OFF)
  elseif(RKNN_MODEL_ZOO_ROOT)
    set(OpenCV_DIR
        "${RKNN_MODEL_ZOO_ROOT}/3rdparty/opencv/opencv-android-sdk-build/sdk/native/jni/abi-${CMAKE_ANDROID_ARCH_ABI}"
        CACHE PATH "OpenCV from rknn_model_zoo (no objdetect)" FORCE)
    set(AV_TSE_FETCH_OPENCV OFF)
  endif()
endif()

if(CMAKE_SYSTEM_NAME STREQUAL "Android")
  find_package(OpenCV QUIET COMPONENTS core imgproc objdetect imgcodecs)
  if(NOT OpenCV_FOUND AND NOT AV_TSE_BUILD_LIBRARY)
    find_package(OpenCV QUIET COMPONENTS core imgproc)
  endif()
else()
  find_package(OpenCV QUIET COMPONENTS core imgproc objdetect videoio imgcodecs)
endif()
if(NOT OpenCV_FOUND AND AV_TSE_FETCH_OPENCV)
  message(STATUS "av_tse: OpenCV not found; fetching minimal build (may take several minutes)")
  set(BUILD_LIST core,imgproc,objdetect,videoio,imgcodecs CACHE STRING "" FORCE)
  set(BUILD_TESTS OFF CACHE BOOL "" FORCE)
  set(BUILD_PERF_TESTS OFF CACHE BOOL "" FORCE)
  set(BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
  set(BUILD_opencv_apps OFF CACHE BOOL "" FORCE)
  set(BUILD_opencv_world OFF CACHE BOOL "" FORCE)
  set(WITH_FFMPEG OFF CACHE BOOL "" FORCE)
  set(WITH_GTK OFF CACHE BOOL "" FORCE)
  set(WITH_QT OFF CACHE BOOL "" FORCE)
  set(WITH_OPENEXR OFF CACHE BOOL "" FORCE)
  set(WITH_IPP OFF CACHE BOOL "" FORCE)
  set(WITH_ITT OFF CACHE BOOL "" FORCE)
  FetchContent_Declare(
    opencv_fc
    GIT_REPOSITORY https://github.com/opencv/opencv.git
    GIT_TAG 4.8.0)
  FetchContent_MakeAvailable(opencv_fc)
  FetchContent_GetProperties(opencv_fc SOURCE_DIR _opencv_src BINARY_DIR _opencv_bin)
  set(OpenCV_LIBS opencv_core opencv_imgproc opencv_objdetect opencv_videoio opencv_imgcodecs)
  get_filename_component(_cpp_top_build "${CMAKE_CURRENT_BINARY_DIR}/.." ABSOLUTE)
  set(_av_tse_opencv_include_dirs
      "${_opencv_src}/modules/core/include"
      "${_opencv_src}/modules/imgproc/include"
      "${_opencv_src}/modules/objdetect/include"
      "${_opencv_src}/modules/videoio/include"
      "${_opencv_src}/modules/imgcodecs/include"
      "${_opencv_src}/modules/features2d/include"
      "${_opencv_src}/modules/calib3d/include"
      "${_opencv_bin}"
      "${_opencv_bin}/modules/core"
      "${_cpp_top_build}")
endif()
if(NOT OpenCV_FOUND AND NOT TARGET opencv_core)
  message(FATAL_ERROR "OpenCV not found. Install libopencv-dev, set -DOpenCV_DIR=..., or on Android set -DAV_TSE_ANDROID_OPENCV_DIR=... (see scripts/build_opencv_android.sh).")
endif()
if(AV_TSE_BUILD_LIBRARY)
  set(_av_tse_has_objdetect 0)
  if(TARGET opencv_objdetect)
    set(_av_tse_has_objdetect 1)
  elseif(DEFINED OpenCV_LIBS)
    string(FIND "${OpenCV_LIBS}" "opencv_objdetect" _av_objdetect_idx)
    if(NOT _av_objdetect_idx EQUAL -1)
      set(_av_tse_has_objdetect 1)
    endif()
  endif()
  if(NOT _av_tse_has_objdetect)
    message(
      FATAL_ERROR
        "Full libav_tse requires OpenCV objdetect. On Android use scripts/build_opencv_android.sh and -DAV_TSE_ANDROID_OPENCV_DIR=.../sdk/native/jni/abi-arm64-v8a"
    )
  endif()
endif()
if(TARGET opencv_core AND NOT OpenCV_LIBS)
  set(OpenCV_LIBS opencv_core opencv_imgproc opencv_objdetect)
  if(TARGET opencv_imgcodecs)
    list(APPEND OpenCV_LIBS opencv_imgcodecs)
  endif()
  if(TARGET opencv_videoio)
    list(APPEND OpenCV_LIBS opencv_videoio)
  endif()
endif()
if(TARGET opencv_core AND NOT _av_tse_opencv_include_dirs)
  FetchContent_GetProperties(opencv_fc SOURCE_DIR _opencv_src BINARY_DIR _opencv_bin)
  get_filename_component(_cpp_top_build "${CMAKE_CURRENT_BINARY_DIR}/.." ABSOLUTE)
  set(_av_tse_opencv_include_dirs
      "${_opencv_src}/modules/core/include"
      "${_opencv_src}/modules/imgproc/include"
      "${_opencv_src}/modules/objdetect/include"
      "${_opencv_src}/modules/videoio/include"
      "${_opencv_src}/modules/imgcodecs/include"
      "${_opencv_src}/modules/features2d/include"
      "${_opencv_src}/modules/calib3d/include"
      "${_opencv_bin}"
      "${_opencv_bin}/modules/core"
      "${_cpp_top_build}")
endif()

if(NOT AV_TSE_BUILD_LIBRARY)
  # Smoke-only / model-only targets skip heavy deps.
else()
# libsoxr (same strategy as asr_frontend)
find_package(PkgConfig QUIET)
set(_av_tse_soxr "")
if(PkgConfig_FOUND)
  pkg_check_modules(SOXR IMPORTED_TARGET soxr)
  if(TARGET PkgConfig::SOXR)
    set(_av_tse_soxr PkgConfig::SOXR)
  endif()
endif()
if(NOT _av_tse_soxr)
  find_path(SOXR_INCLUDE_DIR NAMES soxr.h)
  find_library(SOXR_LIBRARY NAMES soxr)
  if(SOXR_INCLUDE_DIR AND SOXR_LIBRARY)
    add_library(av_tse_soxr UNKNOWN IMPORTED)
    set_target_properties(
      av_tse_soxr PROPERTIES IMPORTED_LOCATION "${SOXR_LIBRARY}"
                             INTERFACE_INCLUDE_DIRECTORIES "${SOXR_INCLUDE_DIR}")
    set(_av_tse_soxr av_tse_soxr)
  endif()
endif()

set(_av_tse_soxr_bundled FALSE)
if(NOT _av_tse_soxr)
  include(ExternalProject)
  FetchContent_Declare(
    libsoxr_fc
    GIT_REPOSITORY https://github.com/chirlu/soxr.git
    GIT_TAG 0.1.3)
  FetchContent_GetProperties(libsoxr_fc)
  if(NOT libsoxr_fc_POPULATED)
    FetchContent_Populate(libsoxr_fc)
  endif()
  set(_av_soxr_build "${CMAKE_CURRENT_BINARY_DIR}/libsoxr-build")
  set(_av_soxr_lib "${_av_soxr_build}/src/libsoxr.a")
  set(_av_soxr_ep_cmake_args
      -DCMAKE_BUILD_TYPE=Release
      -DCMAKE_POSITION_INDEPENDENT_CODE=ON
      -DBUILD_TESTS=OFF
      -DBUILD_EXAMPLES=OFF
      -DBUILD_SHARED_LIBS=OFF
      -DWITH_OPENMP=OFF
      -DCMAKE_POLICY_VERSION_MINIMUM=3.5)
  if(CMAKE_ANDROID_NDK OR CMAKE_SYSTEM_NAME STREQUAL "Android" OR ANDROID)
    if(CMAKE_TOOLCHAIN_FILE)
      list(APPEND _av_soxr_ep_cmake_args -DCMAKE_TOOLCHAIN_FILE=${CMAKE_TOOLCHAIN_FILE})
    endif()
    if(ANDROID_ABI)
      list(APPEND _av_soxr_ep_cmake_args -DANDROID_ABI=${ANDROID_ABI})
    endif()
    if(ANDROID_PLATFORM)
      list(APPEND _av_soxr_ep_cmake_args -DANDROID_PLATFORM=${ANDROID_PLATFORM})
    endif()
    if(ANDROID_STL)
      list(APPEND _av_soxr_ep_cmake_args -DANDROID_STL=${ANDROID_STL})
    endif()
    if(CMAKE_ANDROID_NDK)
      list(APPEND _av_soxr_ep_cmake_args -DCMAKE_ANDROID_NDK=${CMAKE_ANDROID_NDK})
    endif()
  endif()
  ExternalProject_Add(
    av_tse_libsoxr_ext
    SOURCE_DIR "${libsoxr_fc_SOURCE_DIR}"
    BINARY_DIR "${_av_soxr_build}"
    DOWNLOAD_COMMAND ""
    UPDATE_COMMAND ""
    CMAKE_ARGS ${_av_soxr_ep_cmake_args}
    BUILD_BYPRODUCTS "${_av_soxr_lib}"
    INSTALL_COMMAND "")
  add_library(av_tse_bundled_soxr STATIC IMPORTED)
  set_target_properties(
    av_tse_bundled_soxr PROPERTIES IMPORTED_LOCATION "${_av_soxr_lib}"
                                   INTERFACE_INCLUDE_DIRECTORIES "${libsoxr_fc_SOURCE_DIR}/src")
  set(_av_tse_soxr av_tse_bundled_soxr)
  set(_av_tse_soxr_bundled TRUE)
endif()

# yaml-cpp
FetchContent_Declare(
  yaml-cpp
  GIT_REPOSITORY https://github.com/jbeder/yaml-cpp.git
  GIT_TAG 0.8.0)
set(YAML_CPP_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(YAML_CPP_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(yaml-cpp)

endif()  # AV_TSE_BUILD_LIBRARY

set(_av_tse_inference_libs "")
set(_av_tse_private_includes "")
set(_av_rknn_lib_dir "")
set(_av_ort_lib_dir "")

if(AV_TSE_INFERENCE_BACKEND STREQUAL "RKNN")
  list(APPEND _av_tse_private_includes "${CMAKE_CURRENT_SOURCE_DIR}/../asr_frontend/include")

  if(NOT DEFINED RKNN_RKNPU2_ROOT AND DEFINED ENV{RKNN_RKNPU2_ROOT})
    set(RKNN_RKNPU2_ROOT "$ENV{RKNN_RKNPU2_ROOT}")
  endif()
  if(NOT DEFINED RKNN_MODEL_ZOO_ROOT AND DEFINED ENV{RKNN_MODEL_ZOO_ROOT})
    set(RKNN_MODEL_ZOO_ROOT "$ENV{RKNN_MODEL_ZOO_ROOT}")
  endif()
  if(RKNN_RKNPU2_ROOT)
    set(_rknn_root "${RKNN_RKNPU2_ROOT}")
  elseif(RKNN_MODEL_ZOO_ROOT)
    set(_rknn_root "${RKNN_MODEL_ZOO_ROOT}/3rdparty/rknpu2")
  else()
    message(
      FATAL_ERROR
        "RKNN backend needs either -DRKNN_RKNPU2_ROOT=/path/to/rknpu2 (include + Android/<abi>/librknnrt.so) "
        "or -DRKNN_MODEL_ZOO_ROOT=/path/to/rknn_model_zoo (uses .../3rdparty/rknpu2).")
  endif()
  if(CMAKE_SYSTEM_NAME STREQUAL "Android")
    set(_rknn_arch "${CMAKE_ANDROID_ARCH_ABI}")
  else()
    if(CMAKE_SIZEOF_VOID_P EQUAL 8)
      set(_rknn_arch aarch64)
    else()
      set(_rknn_arch armhf)
    endif()
  endif()
  set(_rknn_include "${_rknn_root}/include")
  set(_rknn_lib "${_rknn_root}/${CMAKE_SYSTEM_NAME}/${_rknn_arch}/librknnrt.so")
  # Alternative layout: rknn-toolkit2 rknpu2/runtime/{Android,Linux}/librknn_api/{include,<abi>}
  if(NOT EXISTS "${_rknn_include}/rknn_api.h")
    set(_rknn_rt_android "${_rknn_root}/runtime/Android/librknn_api")
    set(_rknn_rt_linux "${_rknn_root}/runtime/Linux/librknn_api")
    if(CMAKE_SYSTEM_NAME STREQUAL "Android" AND EXISTS "${_rknn_rt_android}/include/rknn_api.h")
      set(_rknn_include "${_rknn_rt_android}/include")
      set(_rknn_lib "${_rknn_rt_android}/${_rknn_arch}/librknnrt.so")
    elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND EXISTS "${_rknn_rt_linux}/include/rknn_api.h")
      set(_rknn_include "${_rknn_rt_linux}/include")
      set(_rknn_lib "${_rknn_rt_linux}/${_rknn_arch}/librknnrt.so")
    endif()
  endif()
  if(NOT EXISTS "${_rknn_include}/rknn_api.h")
    message(
      FATAL_ERROR
        "RKNN headers not found under ${_rknn_include}. "
        "Set RKNN_RKNPU2_ROOT to the rknpu2 SDK root (folder containing include/rknn_api.h), "
        "or RKNN_MODEL_ZOO_ROOT to a full rknn_model_zoo checkout.")
  endif()
  if(NOT EXISTS "${_rknn_lib}")
    message(
      FATAL_ERROR
        "librknnrt.so not found: ${_rknn_lib}. "
        "Expected <rknpu2>/${CMAKE_SYSTEM_NAME}/${_rknn_arch}/librknnrt.so (rknn_model_zoo) "
        "or <rknpu2>/runtime/${CMAKE_SYSTEM_NAME}/librknn_api/${_rknn_arch}/librknnrt.so (rknn-toolkit2).")
  endif()
  list(APPEND _av_tse_private_includes "${_rknn_include}")
  list(APPEND _av_tse_inference_libs "${_rknn_lib}")
  get_filename_component(_av_rknn_lib_dir "${_rknn_lib}" DIRECTORY)
else()
  if(NOT DEFINED ONNXRUNTIME_ROOT AND DEFINED ENV{ONNXRUNTIME_ROOT})
    set(ONNXRUNTIME_ROOT "$ENV{ONNXRUNTIME_ROOT}")
  endif()
  if(NOT ONNXRUNTIME_ROOT)
    get_property(_cached_ort_root CACHE ONNXRUNTIME_ROOT PROPERTY VALUE)
    if(_cached_ort_root)
      set(ONNXRUNTIME_ROOT "${_cached_ort_root}")
    endif()
  endif()
  if(NOT ONNXRUNTIME_ROOT AND DEFINED ASR_FRONTEND_ONNXRUNTIME_LIB_DIR)
    get_filename_component(ONNXRUNTIME_ROOT "${ASR_FRONTEND_ONNXRUNTIME_LIB_DIR}/.." ABSOLUTE)
  endif()
  set(_ort_ver "1.17.1")
  if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    set(_av_ort_cache_root "${CMAKE_CURRENT_BINARY_DIR}")
  else()
    get_filename_component(_av_ort_cache_root "${CMAKE_CURRENT_BINARY_DIR}/.." ABSOLUTE)
  endif()
  set(_av_ort_android_abi "${CMAKE_ANDROID_ARCH_ABI}")
  if(NOT _av_ort_android_abi)
    set(_av_ort_android_abi "${ANDROID_ABI}")
  endif()
  if(NOT _av_ort_android_abi)
    set(_av_ort_android_abi "arm64-v8a")
  endif()
  set(_av_is_android FALSE)
  if(CMAKE_ANDROID_NDK OR CMAKE_SYSTEM_NAME STREQUAL "Android")
    set(_av_is_android TRUE)
  endif()
  if(NOT ONNXRUNTIME_ROOT
     AND EXISTS "${_av_ort_cache_root}/onnxruntime-android-${_ort_ver}/headers/onnxruntime_cxx_api.h")
    set(ONNXRUNTIME_ROOT "${_av_ort_cache_root}/onnxruntime-android-${_ort_ver}")
  endif()
  if(NOT ONNXRUNTIME_ROOT)
    if(_av_is_android AND _av_ort_android_abi STREQUAL "arm64-v8a")
      set(ONNXRUNTIME_ROOT "${_av_ort_cache_root}/onnxruntime-android-${_ort_ver}")
      if(NOT EXISTS "${ONNXRUNTIME_ROOT}/headers/onnxruntime_cxx_api.h")
        set(_ort_aar "${_av_ort_cache_root}/onnxruntime-android-${_ort_ver}.aar")
        message(STATUS "Downloading ONNX Runtime Android ${_ort_ver} AAR for av_tse")
        file(
          DOWNLOAD
          "https://repo1.maven.org/maven2/com/microsoft/onnxruntime/onnxruntime-android/${_ort_ver}/onnxruntime-android-${_ort_ver}.aar"
          "${_ort_aar}"
          SHOW_PROGRESS)
        file(MAKE_DIRECTORY "${ONNXRUNTIME_ROOT}")
        execute_process(
          COMMAND ${CMAKE_COMMAND} -E tar xf "${_ort_aar}"
          WORKING_DIRECTORY "${ONNXRUNTIME_ROOT}"
          COMMAND_ERROR_IS_FATAL ANY)
      endif()
    elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux")
      set(ONNXRUNTIME_ROOT "${_av_ort_cache_root}/onnxruntime-linux-x64-${_ort_ver}")
      if(NOT EXISTS "${ONNXRUNTIME_ROOT}/include/onnxruntime_cxx_api.h")
        set(_ort_tgz "${_av_ort_cache_root}/onnxruntime-linux-x64-${_ort_ver}.tgz")
        if(NOT EXISTS "${_ort_tgz}")
          message(STATUS "Downloading ONNX Runtime ${_ort_ver} for av_tse")
          file(
            DOWNLOAD
            "https://github.com/microsoft/onnxruntime/releases/download/v${_ort_ver}/onnxruntime-linux-x64-${_ort_ver}.tgz"
            "${_ort_tgz}"
            SHOW_PROGRESS)
          execute_process(
            COMMAND ${CMAKE_COMMAND} -E tar xzf "${_ort_tgz}"
            WORKING_DIRECTORY "${_av_ort_cache_root}"
            COMMAND_ERROR_IS_FATAL ANY)
        endif()
      endif()
    endif()
  endif()
  set(_ort_lib_hint_dirs "")
  if(ONNXRUNTIME_ROOT)
    if(EXISTS "${ONNXRUNTIME_ROOT}/include/onnxruntime_cxx_api.h")
      set(ONNXRUNTIME_INCLUDE_DIR "${ONNXRUNTIME_ROOT}/include")
      list(APPEND _ort_lib_hint_dirs "${ONNXRUNTIME_ROOT}/lib")
    elseif(EXISTS "${ONNXRUNTIME_ROOT}/headers/onnxruntime_cxx_api.h")
      set(ONNXRUNTIME_INCLUDE_DIR "${ONNXRUNTIME_ROOT}/headers")
      list(APPEND _ort_lib_hint_dirs "${ONNXRUNTIME_ROOT}/jni/${_av_ort_android_abi}")
    endif()
    set(_ort_so "${ONNXRUNTIME_ROOT}/jni/${_av_ort_android_abi}/libonnxruntime.so")
    if(EXISTS "${_ort_so}")
      set(ONNXRUNTIME_LIBRARY "${_ort_so}")
    else()
      find_library(ONNXRUNTIME_LIBRARY onnxruntime HINTS ${_ort_lib_hint_dirs} NO_DEFAULT_PATH)
    endif()
  else()
    find_path(ONNXRUNTIME_INCLUDE_DIR NAMES onnxruntime_cxx_api.h)
    find_library(ONNXRUNTIME_LIBRARY NAMES onnxruntime)
  endif()
  if(NOT ONNXRUNTIME_INCLUDE_DIR OR NOT ONNXRUNTIME_LIBRARY)
    message(FATAL_ERROR "ONNX Runtime not found for av_tse. Set ONNXRUNTIME_ROOT.")
  endif()
  list(APPEND _av_tse_private_includes "${ONNXRUNTIME_INCLUDE_DIR}")
  list(APPEND _av_tse_inference_libs "${ONNXRUNTIME_LIBRARY}")
  get_filename_component(_av_ort_lib_dir "${ONNXRUNTIME_LIBRARY}" DIRECTORY)
  set(AV_TSE_ONNXRUNTIME_LIB_DIR "${_av_ort_lib_dir}" PARENT_SCOPE)
endif()

set(_av_tse_model_sources "")
if(AV_TSE_INFERENCE_BACKEND STREQUAL "RKNN")
  list(APPEND _av_tse_model_sources src/av_mossformer_rknn.cpp ../asr_frontend/src/rknn_session.cpp)
else()
  list(APPEND _av_tse_model_sources src/av_mossformer_onnx.cpp)
endif()

function(av_tse_configure_opencv_target target_name)
  if(_av_tse_opencv_include_dirs)
    target_include_directories(${target_name} PUBLIC ${_av_tse_opencv_include_dirs})
  endif()
  if(TARGET opencv_core)
    get_filename_component(_cpp_top_build "${CMAKE_CURRENT_BINARY_DIR}/.." ABSOLUTE)
    target_include_directories(${target_name} PUBLIC "${_cpp_top_build}")
  endif()
  if(TARGET opencv_core)
    set(_libs opencv_core opencv_imgproc)
    if(TARGET opencv_objdetect)
      list(APPEND _libs opencv_objdetect)
    endif()
    if(TARGET opencv_videoio)
      list(APPEND _libs opencv_videoio)
    endif()
    if(TARGET opencv_imgcodecs)
      list(APPEND _libs opencv_imgcodecs)
    endif()
    target_link_libraries(${target_name} PUBLIC ${_libs})
    add_dependencies(${target_name} opencv_core opencv_imgproc)
  else()
    target_link_libraries(${target_name} PUBLIC ${OpenCV_LIBS})
    target_include_directories(${target_name} PUBLIC ${OpenCV_INCLUDE_DIRS})
  endif()
endfunction()

if(AV_TSE_BUILD_LIBRARY)
  set(_av_tse_sources
      src/av_tse_config.cpp
      src/audio_resampler.cpp
      src/face_haar_tracker.cpp
      src/incremental_video_resampler.cpp
      src/av_align.cpp
      src/av_stream_inference.cpp
      src/stream_inference_sdk.cpp
      ${_av_tse_model_sources})

  add_library(av_tse STATIC ${_av_tse_sources})
  target_include_directories(av_tse PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
                           PRIVATE ${_av_tse_private_includes})
  av_tse_configure_opencv_target(av_tse)

  if(_opencv_src)
    set(_av_tse_haar "${_opencv_src}/data/haarcascades/haarcascade_frontalface_default.xml")
    if(EXISTS "${_av_tse_haar}")
      target_compile_definitions(av_tse PRIVATE AV_TSE_HAAR_CASCADE_PATH="${_av_tse_haar}")
    endif()
  elseif(CMAKE_SYSTEM_NAME STREQUAL "Android" AND AV_TSE_ANDROID_OPENCV_DIR)
    get_filename_component(
      _av_tse_haar_android
      "${AV_TSE_ANDROID_OPENCV_DIR}/../../../etc/haarcascades/haarcascade_frontalface_default.xml"
      ABSOLUTE)
    if(EXISTS "${_av_tse_haar_android}")
      target_compile_definitions(av_tse PRIVATE AV_TSE_HAAR_CASCADE_PATH="${_av_tse_haar_android}")
    endif()
  endif()

  target_link_libraries(av_tse PUBLIC Eigen3::Eigen PRIVATE ${_av_tse_soxr} yaml-cpp::yaml-cpp
                        ${_av_tse_inference_libs})
  if(AV_TSE_INFERENCE_BACKEND STREQUAL "RKNN")
    target_compile_definitions(av_tse PUBLIC AV_TSE_USE_RKNN=1)
    set(AV_TSE_RKNN_LIB_DIR "${_av_rknn_lib_dir}" PARENT_SCOPE)
  endif()
  if(_av_tse_soxr_bundled)
    add_dependencies(av_tse av_tse_libsoxr_ext)
  endif()
  if(UNIX AND NOT APPLE)
    if(CMAKE_SYSTEM_NAME STREQUAL "Android")
      target_link_libraries(av_tse PRIVATE dl log)
    else()
      target_link_libraries(av_tse PRIVATE dl pthread)
    endif()
  endif()
endif()

if(TARGET opencv_videoio)
  set(_av_tse_have_videoio 1)
endif()

if(AV_TSE_BUILD_CASEB)
  if(NOT AV_TSE_BUILD_LIBRARY)
    message(FATAL_ERROR "AV_TSE_BUILD_CASEB requires AV_TSE_BUILD_LIBRARY=ON")
  endif()
  add_executable(av_tse_caseb examples/caseb_main.cpp src/caseb_common.cpp)
  target_include_directories(av_tse_caseb PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
  target_compile_definitions(
    av_tse_caseb PRIVATE AV_TSE_CASEB_DATA_ROOT="${AV_TSE_CASEB_DATA_ROOT}")
  if(_av_tse_have_videoio)
    target_compile_definitions(av_tse_caseb PRIVATE AV_TSE_HAVE_VIDEOIO=1)
  endif()
  target_link_libraries(av_tse_caseb PRIVATE av_tse)
  av_tse_configure_opencv_target(av_tse_caseb)
  if(AV_TSE_INFERENCE_BACKEND STREQUAL "RKNN")
    set_target_properties(av_tse_caseb PROPERTIES BUILD_RPATH "${_av_rknn_lib_dir}")
  elseif(DEFINED _av_ort_lib_dir)
    set_target_properties(av_tse_caseb PROPERTIES BUILD_RPATH "${_av_ort_lib_dir}")
  endif()
  if(UNIX AND NOT APPLE)
    if(CMAKE_SYSTEM_NAME STREQUAL "Android")
      target_link_libraries(av_tse_caseb PRIVATE dl log)
    else()
      target_link_libraries(av_tse_caseb PRIVATE dl pthread)
    endif()
  endif()
endif()

if(AV_TSE_BUILD_SMOKE AND AV_TSE_INFERENCE_BACKEND STREQUAL "RKNN")
  add_executable(av_tse_smoke examples/smoke_main.cpp ${_av_tse_model_sources})
  target_include_directories(
    av_tse_smoke
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR}/../asr_frontend/include
            ${_av_tse_private_includes})
  target_compile_definitions(av_tse_smoke PRIVATE AV_TSE_USE_RKNN=1)
  target_link_libraries(av_tse_smoke PRIVATE Eigen3::Eigen ${_av_tse_inference_libs})
  if(TARGET opencv_core)
    target_link_libraries(av_tse_smoke PRIVATE opencv_core opencv_imgproc)
  endif()
  if(UNIX AND NOT APPLE)
    if(CMAKE_SYSTEM_NAME STREQUAL "Android")
      target_link_libraries(av_tse_smoke PRIVATE dl log)
    else()
      target_link_libraries(av_tse_smoke PRIVATE dl pthread)
    endif()
  endif()
  set_target_properties(av_tse_smoke PROPERTIES BUILD_RPATH "${_av_rknn_lib_dir}")
endif()
