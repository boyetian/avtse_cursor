#include "av_tse/av_tse_config.hpp"

#include <yaml-cpp/yaml.h>

#include <stdexcept>

namespace av_tse {

AvTseConfig load_av_tse_config(const std::string& yaml_path) {
  YAML::Node root = YAML::LoadFile(yaml_path);
  AvTseConfig cfg;
  if (root["audio_sr"]) {
    cfg.audio_sr = root["audio_sr"].as<int>();
  }
  if (root["ref_sr"]) {
    cfg.ref_sr = root["ref_sr"].as<float>();
  }
  const YAML::Node na = root["network_audio"];
  if (!na) {
    throw std::runtime_error("config missing network_audio");
  }
  if (na["backbone"]) {
    cfg.backbone = na["backbone"].as<std::string>();
  }
  if (na["image_size"]) {
    cfg.image_size = na["image_size"].as<int>();
  }
  if (na["encoder_kernel_size"]) {
    cfg.encoder_kernel_size = na["encoder_kernel_size"].as<int>();
  }
  if (cfg.backbone != "av_mossformer2_tse" && cfg.backbone != "av_skim") {
    throw std::runtime_error("unsupported backbone: " + cfg.backbone);
  }
  return cfg;
}

}  // namespace av_tse
