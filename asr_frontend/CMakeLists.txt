cmake_minimum_required(VERSION 3.16)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(ASR_FRONTEND_BUILD_TESTS "Build unit tests (needs GTest)" ON)
option(ASR_FRONTEND_DOWNLOAD_ONNXRUNTIME
       "Download ONNX Runtime (Linux x64 tarball or Android arm64 AAR) if ONNXRUNTIME_ROOT is unset"
       ON)

set(ASR_FRONTEND_INFERENCE_BACKEND "ONNX" CACHE STRING "Inference backend: ONNX or RKNN")
set_property(CACHE ASR_FRONTEND_INFERENCE_BACKEND PROPERTY STRINGS ONNX RKNN)

if(NOT ASR_FRONTEND_INFERENCE_BACKEND MATCHES "^(ONNX|RKNN)$")
  message(FATAL_ERROR "ASR_FRONTEND_INFERENCE_BACKEND must be ONNX or RKNN")
endif()

include(FetchContent)

find_package(Eigen3 QUIET CONFIG)
if(NOT Eigen3_FOUND)
  FetchContent_Declare(
    eigen
    GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
    GIT_TAG 3.4.0
  )
  set(EIGEN_BUILD_DOC OFF CACHE BOOL "" FORCE)
  set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(eigen)
endif()

# libsoxr (resampling)
find_package(PkgConfig QUIET)
set(_asr_frontend_soxr "")
if(PkgConfig_FOUND)
  pkg_check_modules(SOXR IMPORTED_TARGET soxr)
  if(TARGET PkgConfig::SOXR)
    set(_asr_frontend_soxr PkgConfig::SOXR)
  endif()
endif()
if(NOT _asr_frontend_soxr)
  find_path(SOXR_INCLUDE_DIR NAMES soxr.h)
  find_library(SOXR_LIBRARY NAMES soxr)
  if(SOXR_INCLUDE_DIR AND SOXR_LIBRARY)
    add_library(asr_frontend_soxr UNKNOWN IMPORTED)
    set_target_properties(
      asr_frontend_soxr PROPERTIES IMPORTED_LOCATION "${SOXR_LIBRARY}"
                                 INTERFACE_INCLUDE_DIRECTORIES "${SOXR_INCLUDE_DIR}")
    set(_asr_frontend_soxr asr_frontend_soxr)
  endif()
endif()

set(_asr_frontend_soxr_bundled FALSE)
if(NOT _asr_frontend_soxr)
  include(ExternalProject)
  set(_asr_soxr_build "${CMAKE_CURRENT_BINARY_DIR}/libsoxr-build")
  if(MSVC)
    set(_asr_soxr_lib "${_asr_soxr_build}/src/${CMAKE_CFG_INTDIR}/soxr.lib")
  else()
    set(_asr_soxr_lib "${_asr_soxr_build}/src/libsoxr.a")
  endif()

  FetchContent_Declare(
    libsoxr_fc
    GIT_REPOSITORY https://github.com/chirlu/soxr.git
    GIT_TAG 0.1.3)
  FetchContent_GetProperties(libsoxr_fc)
  if(NOT libsoxr_fc_POPULATED)
    FetchContent_Populate(libsoxr_fc)
  endif()
  set(_asr_soxr_src "${libsoxr_fc_SOURCE_DIR}")

  set(_asr_soxr_ep_cmake_args
      -DCMAKE_BUILD_TYPE=Release
      -DCMAKE_POSITION_INDEPENDENT_CODE=ON
      -DBUILD_TESTS=OFF
      -DBUILD_EXAMPLES=OFF
      -DBUILD_SHARED_LIBS=OFF
      -DWITH_OPENMP=OFF
      -DCMAKE_POLICY_VERSION_MINIMUM=3.5)
  if(CMAKE_ANDROID_NDK OR CMAKE_SYSTEM_NAME STREQUAL "Android" OR ANDROID)
    if(CMAKE_TOOLCHAIN_FILE)
      list(APPEND _asr_soxr_ep_cmake_args -DCMAKE_TOOLCHAIN_FILE=${CMAKE_TOOLCHAIN_FILE})
    endif()
    if(ANDROID_ABI)
      list(APPEND _asr_soxr_ep_cmake_args -DANDROID_ABI=${ANDROID_ABI})
    endif()
    if(ANDROID_PLATFORM)
      list(APPEND _asr_soxr_ep_cmake_args -DANDROID_PLATFORM=${ANDROID_PLATFORM})
    endif()
    if(ANDROID_STL)
      list(APPEND _asr_soxr_ep_cmake_args -DANDROID_STL=${ANDROID_STL})
    endif()
    if(CMAKE_ANDROID_NDK)
      list(APPEND _asr_soxr_ep_cmake_args -DCMAKE_ANDROID_NDK=${CMAKE_ANDROID_NDK})
    endif()
  endif()

  ExternalProject_Add(
    asr_frontend_libsoxr_ext
    SOURCE_DIR "${_asr_soxr_src}"
    BINARY_DIR "${_asr_soxr_build}"
    DOWNLOAD_COMMAND ""
    UPDATE_COMMAND ""
    CMAKE_ARGS ${_asr_soxr_ep_cmake_args}
    BUILD_BYPRODUCTS "${_asr_soxr_lib}"
    INSTALL_COMMAND "")
  add_library(asr_frontend_bundled_soxr STATIC IMPORTED)
  set_target_properties(
    asr_frontend_bundled_soxr PROPERTIES IMPORTED_LOCATION "${_asr_soxr_lib}"
                                         INTERFACE_INCLUDE_DIRECTORIES "${_asr_soxr_src}/src")
  set(_asr_frontend_soxr asr_frontend_bundled_soxr)
  set(_asr_frontend_soxr_bundled TRUE)
endif()

set(_asr_frontend_sources
  src/stream_buffer_2d.cpp
  src/audio_resampler.cpp
  src/asr_frontend_api.cpp
  src/stream_inference.cpp
)

set(_asr_frontend_inference_libs "")
set(_asr_frontend_public_includes "${CMAKE_CURRENT_SOURCE_DIR}/include")
set(_asr_rknn_lib_dir "")

if(ASR_FRONTEND_INFERENCE_BACKEND STREQUAL "RKNN")
  list(APPEND _asr_frontend_sources
       src/rknn_session.cpp
       src/dfsmn_aec_rknn.cpp
       src/zipenhancer_full_rknn.cpp)

  if(NOT DEFINED RKNN_MODEL_ZOO_ROOT AND DEFINED ENV{RKNN_MODEL_ZOO_ROOT})
    set(RKNN_MODEL_ZOO_ROOT "$ENV{RKNN_MODEL_ZOO_ROOT}")
  endif()
  if(NOT RKNN_MODEL_ZOO_ROOT)
    message(FATAL_ERROR
            "RKNN backend requires -DRKNN_MODEL_ZOO_ROOT=/path/to/rknn_model_zoo")
  endif()

  set(TARGET_SOC "rk3588" CACHE STRING "Rockchip NPU SoC (rk3588, rk356x, rk3576, ...)")

  set(_rknn_root "${RKNN_MODEL_ZOO_ROOT}/3rdparty/rknpu2")
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
  if(NOT EXISTS "${_rknn_include}/rknn_api.h")
    message(FATAL_ERROR "RKNN headers not found under ${_rknn_include}")
  endif()
  if(NOT EXISTS "${_rknn_lib}")
    message(FATAL_ERROR "librknnrt.so not found: ${_rknn_lib}")
  endif()

  list(APPEND _asr_frontend_public_includes "${_rknn_include}")
  list(APPEND _asr_frontend_inference_libs "${_rknn_lib}")
  get_filename_component(_asr_rknn_lib_dir "${_rknn_lib}" DIRECTORY)

else()
  list(APPEND _asr_frontend_sources
       src/dfsmn_aec_onnx.cpp
       src/zipenhancer_full_onnx.cpp)

  if(NOT DEFINED ONNXRUNTIME_ROOT AND DEFINED ENV{ONNXRUNTIME_ROOT})
    set(ONNXRUNTIME_ROOT "$ENV{ONNXRUNTIME_ROOT}")
  endif()

  set(_ort_ver "1.17.1")
  if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    set(_asr_ort_cache_root "${CMAKE_CURRENT_BINARY_DIR}")
  else()
    get_filename_component(_asr_ort_cache_root "${CMAKE_CURRENT_BINARY_DIR}/.." ABSOLUTE)
  endif()
  set(_asr_ort_android_abi "${CMAKE_ANDROID_ARCH_ABI}")
  if(NOT _asr_ort_android_abi)
    set(_asr_ort_android_abi "${ANDROID_ABI}")
  endif()
  if(NOT _asr_ort_android_abi)
    set(_asr_ort_android_abi "arm64-v8a")
  endif()
  set(_asr_is_android FALSE)
  if(CMAKE_ANDROID_NDK OR CMAKE_SYSTEM_NAME STREQUAL "Android")
    set(_asr_is_android TRUE)
  endif()
  if(NOT ONNXRUNTIME_ROOT
     AND EXISTS "${_asr_ort_cache_root}/onnxruntime-android-${_ort_ver}/headers/onnxruntime_cxx_api.h")
    set(ONNXRUNTIME_ROOT "${_asr_ort_cache_root}/onnxruntime-android-${_ort_ver}")
  endif()
  if(NOT ONNXRUNTIME_ROOT AND ASR_FRONTEND_DOWNLOAD_ONNXRUNTIME)
    if(_asr_is_android AND _asr_ort_android_abi STREQUAL "arm64-v8a")
      set(ONNXRUNTIME_ROOT "${_asr_ort_cache_root}/onnxruntime-android-${_ort_ver}")
      if(NOT EXISTS "${ONNXRUNTIME_ROOT}/headers/onnxruntime_cxx_api.h")
        set(_ort_aar "${_asr_ort_cache_root}/onnxruntime-android-${_ort_ver}.aar")
        message(STATUS "Downloading ONNX Runtime Android ${_ort_ver} AAR to ${_ort_aar}")
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
    elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux"
           AND (CMAKE_HOST_SYSTEM_PROCESSOR STREQUAL "x86_64" OR CMAKE_HOST_SYSTEM_PROCESSOR STREQUAL "AMD64"))
      set(ONNXRUNTIME_ROOT "${_asr_ort_cache_root}/onnxruntime-linux-x64-${_ort_ver}")
      if(NOT EXISTS "${ONNXRUNTIME_ROOT}/include/onnxruntime_cxx_api.h")
        set(_ort_tgz "${_asr_ort_cache_root}/onnxruntime-linux-x64-${_ort_ver}.tgz")
        message(STATUS "Downloading ONNX Runtime ${_ort_ver} to ${_ort_tgz}")
        file(
          DOWNLOAD
          "https://github.com/microsoft/onnxruntime/releases/download/v${_ort_ver}/onnxruntime-linux-x64-${_ort_ver}.tgz"
          "${_ort_tgz}"
          SHOW_PROGRESS)
        execute_process(
          COMMAND ${CMAKE_COMMAND} -E tar xzf "${_ort_tgz}"
          WORKING_DIRECTORY "${_asr_ort_cache_root}"
          COMMAND_ERROR_IS_FATAL ANY)
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
      list(APPEND _ort_lib_hint_dirs "${ONNXRUNTIME_ROOT}/jni/${_asr_ort_android_abi}")
    endif()
    set(_ort_so "${ONNXRUNTIME_ROOT}/jni/${_asr_ort_android_abi}/libonnxruntime.so")
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
    message(
      FATAL_ERROR
        "ONNX Runtime not found. Set -DONNXRUNTIME_ROOT=..., or enable ASR_FRONTEND_DOWNLOAD_ONNXRUNTIME on Linux x64 / Android arm64-v8a."
    )
  endif()

  list(APPEND _asr_frontend_public_includes "${ONNXRUNTIME_INCLUDE_DIR}")
  list(APPEND _asr_frontend_inference_libs "${ONNXRUNTIME_LIBRARY}")
  get_filename_component(_asr_ort_lib_dir "${ONNXRUNTIME_LIBRARY}" DIRECTORY)
  set(ASR_FRONTEND_ONNXRUNTIME_LIB_DIR "${_asr_ort_lib_dir}" PARENT_SCOPE)
  set(ONNXRUNTIME_ROOT "${ONNXRUNTIME_ROOT}" CACHE PATH "ONNX Runtime root (shared by av_tse)" FORCE)
  set(ASR_FRONTEND_ONNXRUNTIME_LIB_DIR "${_asr_ort_lib_dir}" CACHE INTERNAL "ONNX Runtime lib dir" FORCE)
endif()

add_library(asr_frontend STATIC ${_asr_frontend_sources})
target_include_directories(asr_frontend PUBLIC ${_asr_frontend_public_includes})
target_link_libraries(asr_frontend
  PUBLIC
    Eigen3::Eigen
  PRIVATE
    ${_asr_frontend_soxr}
    ${_asr_frontend_inference_libs}
)

if(ASR_FRONTEND_INFERENCE_BACKEND STREQUAL "RKNN")
  target_compile_definitions(asr_frontend PUBLIC ASR_FRONTEND_USE_RKNN=1)
  set(ASR_FRONTEND_RKNN_LIB_DIR "${_asr_rknn_lib_dir}" PARENT_SCOPE)
endif()

if(_asr_frontend_soxr_bundled)
  add_dependencies(asr_frontend asr_frontend_libsoxr_ext)
endif()
if(UNIX AND NOT APPLE)
  if(CMAKE_SYSTEM_NAME STREQUAL "Android")
    target_link_libraries(asr_frontend PRIVATE dl log)
  else()
    target_link_libraries(asr_frontend PRIVATE dl pthread)
  endif()
endif()

add_executable(asr_frontend_smoke examples/smoke_main.cpp)
target_link_libraries(asr_frontend_smoke PRIVATE asr_frontend)

if(ASR_FRONTEND_INFERENCE_BACKEND STREQUAL "RKNN")
  set_target_properties(asr_frontend_smoke PROPERTIES BUILD_RPATH "${_asr_rknn_lib_dir}")
else()
  set_target_properties(asr_frontend_smoke PROPERTIES BUILD_RPATH "${_asr_ort_lib_dir}")
endif()

if(ASR_FRONTEND_BUILD_TESTS AND ASR_FRONTEND_INFERENCE_BACKEND STREQUAL "ONNX")
  find_package(GTest QUIET)
  if(GTest_FOUND)
    add_executable(asr_frontend_tests tests/test_stream_buffer.cpp)
    target_link_libraries(asr_frontend_tests PRIVATE asr_frontend GTest::gtest_main)
    set_target_properties(asr_frontend_tests PROPERTIES BUILD_RPATH "${_asr_ort_lib_dir}")
    include(GoogleTest)
    gtest_discover_tests(asr_frontend_tests)
  else()
    message(STATUS "GTest not found; skipping asr_frontend_tests")
  endif()
elseif(ASR_FRONTEND_BUILD_TESTS)
  message(STATUS "asr_frontend_tests skipped (RKNN backend)")
endif()

# Install layout for adb push (Android smoke; prefer build_android_rk3588.sh manual packaging)
if((CMAKE_ANDROID_NDK OR CMAKE_SYSTEM_NAME STREQUAL "Android" OR ANDROID) AND TARGET asr_frontend_smoke)
  install(TARGETS asr_frontend_smoke RUNTIME DESTINATION .)
  if(ASR_FRONTEND_INFERENCE_BACKEND STREQUAL "RKNN")
    install(PROGRAMS "${_rknn_lib}" DESTINATION lib)
  else()
    install(PROGRAMS "${ONNXRUNTIME_LIBRARY}" DESTINATION lib)
  endif()
  install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/scripts/README_MODELS.txt"
          DESTINATION model
          OPTIONAL)
endif()
