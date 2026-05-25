#pragma once

#include <string>

namespace av_tse {

enum class CaseBLayout {
  /// audio/test03.wav, video/test03_frames/*.jpg only (no mp4 on device), model/, out/
  AndroidFlat,
  /// 测试用例/音频|视频, checkpoints/AV_Mossformer/ (desktop gtest)
  DesktopRepo,
};

struct CaseBPaths {
  CaseBLayout layout = CaseBLayout::DesktopRepo;
  std::string data_root;
  std::string audio_wav;
  std::string video_mp4;
  std::string video_frames_dir;
  std::string config_yaml;
  std::string model_path;
  std::string output_wav;
};

CaseBPaths resolveCaseBPaths(const std::string& data_root, CaseBLayout layout);

/// Run Case B stream demo. Returns 0 on success, non-zero on error.
int runCaseB(const CaseBPaths& paths);

}  // namespace av_tse
