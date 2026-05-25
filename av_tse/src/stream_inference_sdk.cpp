#include "av_tse/stream_inference_sdk.hpp"

#include <cmath>
#include <stdexcept>

#include "av_tse/av_stream_inference.hpp"

namespace av_tse {

StreamInferenceSDK::StreamInferenceSDK(const StreamInferenceSDKOptions& opt) : opt_(opt) {
  infer_chunk_ms_ = opt.infer_chunk_ms;
  default_fps_ = opt.default_fps;

  AVStreamInferenceOptions core_opt;
  core_opt.config_yaml = opt.config_yaml;
  core_opt.onnx_path = opt.onnx_path;
  core_opt.ref_onnx_path = opt.ref_onnx_path;
  core_opt.rknn_path = opt.rknn_path;
  core_opt.onnx_num_threads = opt.onnx_num_threads;
  core_opt.infer_chunk_ms =
      (opt.core_infer_chunk_ms > 0.f) ? opt.core_infer_chunk_ms : opt.infer_chunk_ms;
  core_opt.context_ms = opt.context_ms;
  core_opt.max_history_ms = opt.max_history_ms;
  core_opt.use_stream_cache = opt.use_stream_cache;
  core_opt.face_tracker = opt.face_tracker;
  core_ = std::make_unique<AVStreamInference>(core_opt);
}

StreamInferenceSDK::~StreamInferenceSDK() { close(); }

int StreamInferenceSDK::audioSr() const { return core_->audioSr(); }

void StreamInferenceSDK::close() {
  if (core_) {
    core_->close();
  }
  audio_buf_.resize(0);
  video_buf_.clear();
  need_core_start_ = true;
}

Eigen::VectorXf StreamInferenceSDK::audioToMono(const Eigen::Ref<const Eigen::MatrixXf>& wav) {
  if (wav.rows() == 1) {
    return wav.row(0);
  }
  return wav.colwise().mean();
}

std::vector<cv::Mat> StreamInferenceSDK::validateFrames(const std::vector<cv::Mat>& frames) {
  if (frames.empty()) {
    throw std::invalid_argument("frames is empty");
  }
  std::vector<cv::Mat> out;
  out.reserve(frames.size());
  for (const auto& f : frames) {
    if (f.empty() || f.channels() != 3) {
      throw std::invalid_argument("frame must be HxWx3 BGR");
    }
    out.push_back(f);
  }
  return out;
}

std::vector<Eigen::VectorXf> StreamInferenceSDK::processAvStream(
    const Eigen::Ref<const Eigen::VectorXf>& audio_mono, const std::vector<cv::Mat>& video_bgr_uint8,
    bool is_start, bool is_end, int sampling_rate, float fps) {
  int sr = sampling_rate;
  if (sr <= 0) {
    throw std::invalid_argument("invalid sampling_rate");
  }
  float fps_f = fps;
  if (!std::isfinite(fps_f) || fps_f <= 1e-3f) {
    fps_f = default_fps_;
  }

  if (is_start) {
    audio_buf_.resize(0);
    video_buf_.clear();
    need_core_start_ = true;
  }

  Eigen::VectorXf a_in = audio_mono;
  std::vector<cv::Mat> v_in = validateFrames(video_bgr_uint8);

  if (a_in.size() > 0) {
    if (audio_buf_.size() == 0) {
      audio_buf_ = a_in;
    } else {
      Eigen::VectorXf cat(audio_buf_.size() + a_in.size());
      cat << audio_buf_, a_in;
      audio_buf_ = std::move(cat);
    }
  }
  if (!v_in.empty()) {
    video_buf_.insert(video_buf_.end(), v_in.begin(), v_in.end());
  }

  const int need_audio =
      std::max(1, static_cast<int>(std::lround(static_cast<float>(sr) * (infer_chunk_ms_ / 1000.f))));
  const int need_video =
      std::max(1, static_cast<int>(std::lround(fps_f * (infer_chunk_ms_ / 1000.f))));

  std::vector<Eigen::VectorXf> outputs_all;

  auto pop_one_block = [&]() {
    Eigen::VectorXf a = audio_buf_.head(need_audio);
    audio_buf_ = audio_buf_.tail(audio_buf_.size() - need_audio);
    std::vector<cv::Mat> v(video_buf_.begin(), video_buf_.begin() + need_video);
    video_buf_.erase(video_buf_.begin(), video_buf_.begin() + need_video);
    return std::make_pair(std::move(a), std::move(v));
  };

  while (audio_buf_.size() >= need_audio &&
         static_cast<int>(video_buf_.size()) >= need_video) {
    auto [a_blk, v_blk] = pop_one_block();
    auto segs = core_->streamInference(a_blk, v_blk, need_core_start_, false, sr);
    need_core_start_ = false;
    for (auto& s : segs) {
      outputs_all.push_back(std::move(s));
    }
  }

  if (is_end) {
    if (audio_buf_.size() > 0 && !video_buf_.empty()) {
      auto segs = core_->streamInference(audio_buf_, video_buf_, need_core_start_, true, sr);
      audio_buf_.resize(0);
      video_buf_.clear();
      need_core_start_ = true;
      for (auto& s : segs) {
        outputs_all.push_back(std::move(s));
      }
    } else {
      audio_buf_.resize(0);
      video_buf_.clear();
      need_core_start_ = true;
    }
  }

  return outputs_all;
}

}  // namespace av_tse
