#pragma once

#include <string>

namespace av_tse {

struct AvTseConfig {
  std::string backbone;
  int audio_sr = 16000;
  float ref_sr = 30.f;
  int image_size = 96;
  int encoder_kernel_size = 16;
};

AvTseConfig load_av_tse_config(const std::string& yaml_path);

}  // namespace av_tse
