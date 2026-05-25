#include "av_tse/av_stream_inference.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>

#include <opencv2/imgproc.hpp>

#include "av_tse/audio_resampler.hpp"
#include "av_tse/av_align.hpp"

namespace av_tse {

namespace {

std::string normalizeModeForBackbone(const std::string& backbone) {
  return (backbone == "av_skim") ? "av_skim" : "mossformer";
}

/// Do not trim more ref frames than needed; keep enough tgt to cover audio_buf after align.
int capRefTrimFrames(int n_drop_ref, int audio_buf_samples, int audio_sr, float ref_sr,
                     int tgt_frame_count) {
  const int min_tgt =
      std::max(1, toVideoIndex(audio_buf_samples, audio_sr, ref_sr) + 2);
  const int max_drop = std::max(0, tgt_frame_count - min_tgt);
  return std::min(std::max(0, n_drop_ref), max_drop);
}

}  // namespace

AVStreamInference::AVStreamInference(const AVStreamInferenceOptions& opt)
    : opt_(opt),
      cfg_(load_av_tse_config(opt.config_yaml)),
      backbone_(cfg_.backbone),
      audio_sr_(cfg_.audio_sr),
      ref_sr_(cfg_.ref_sr),
      image_size_(cfg_.image_size),
      tracker_(opt.face_tracker),
      resampler_(cfg_.ref_sr, cfg_.ref_sr, cfg_.image_size, kMean, kStd,
                 normalizeModeForBackbone(cfg_.backbone)) {
#if defined(AV_TSE_USE_RKNN) && AV_TSE_USE_RKNN
  if (opt_.rknn_path.empty()) {
    throw std::invalid_argument("AVStreamInference requires rknn_path (sep RKNN)");
  }
  if (opt_.ref_onnx_path.empty()) {
    throw std::invalid_argument(
        "AVStreamInference RKNN split deploy requires ref_onnx_path (av_mossformer_ref_fixed.onnx)");
  }
#else
  if (opt_.onnx_path.empty()) {
    throw std::invalid_argument("AVStreamInference requires onnx_path");
  }
#endif

  hop_samples_ = std::max(1, static_cast<int>(std::lround(audio_sr_ * (opt_.infer_chunk_ms / 1000.f))));
  context_samples_ = std::max(0, static_cast<int>(std::lround(audio_sr_ * (opt_.context_ms / 1000.f))));
  lookahead_samples_ =
      std::max(0, static_cast<int>(std::lround(audio_sr_ * (static_cast<float>(opt_.lookahead_frames) / ref_sr_))));
  use_stream_cache_ = opt_.use_stream_cache != 0;
  max_history_samples_ = std::max(hop_samples_ + lookahead_samples_ + 1024,
                                  static_cast<int>(std::lround(audio_sr_ * (opt_.max_history_ms / 1000.f))));
  drop_unit_samples_ = std::max(1, static_cast<int>(std::lround(audio_sr_ * 0.1f)));
  drop_unit_ref_ = std::max(1, static_cast<int>(std::lround(ref_sr_ * 0.1f)));
  const int encoder_stride = std::max(1, cfg_.encoder_kernel_size / 2);
  drop_unit_enc_ = std::max(1, drop_unit_samples_ / encoder_stride);

#if defined(AV_TSE_USE_RKNN) && AV_TSE_USE_RKNN
  model_ = std::make_unique<AvMossformerSplit>(opt_.ref_onnx_path, opt_.rknn_path, 0, 0, image_size_,
                                               opt_.onnx_num_threads);
#else
  model_ = std::make_unique<AvMossformerOnnx>(opt_.onnx_path, opt_.onnx_num_threads);
#endif
  // Match Python: neither ONNX nor RKNN wrapper exposes trim_stream_cache.
  if (use_stream_cache_) {
    use_stream_cache_ = false;
    std::cerr << "[WARN] inference backend has no trim_stream_cache; fallback to use_stream_cache=0\n";
  }
  if (opt_.ingress_warmup != 0) {
    try {
      warmupIngressForward(*model_, image_size_, audio_sr_, ref_sr_, hop_samples_, context_samples_,
                           lookahead_samples_);
    } catch (const std::exception& e) {
      std::cerr << "[WARN] ingress warmup failed: " << e.what() << "\n";
    }
  }

  started_ = false;
  reset();
}

void AVStreamInference::reset() {
  tracker_ = FaceHaarStreamTracker(opt_.face_tracker);
  resampler_ = IncrementalVideoResampler(ref_sr_, ref_sr_, image_size_, kMean, kStd,
                                         normalizeModeForBackbone(backbone_));
  audio_buf_.resize(0);
  produced_samples_ = 0;
  started_ = true;
  model_->clearStreamCache();
}

void AVStreamInference::close() {
  started_ = false;
  model_->clearStreamCache();
}

Eigen::VectorXf AVStreamInference::normalizeAudioChunk(const Eigen::Ref<const Eigen::VectorXf>& audio_chunk,
                                                       int sampling_rate) {
  if (audio_chunk.size() == 0) {
    return Eigen::VectorXf(0);
  }
  Eigen::VectorXf arr = audio_chunk;
  if (sampling_rate != audio_sr_) {
    arr = resample_mono(arr, sampling_rate, audio_sr_);
  }
  return arr;
}

std::vector<cv::Mat> AVStreamInference::normalizeVideoChunk(const std::vector<cv::Mat>& video_chunk) {
  std::vector<cv::Mat> out;
  out.reserve(video_chunk.size());
  for (const auto& f : video_chunk) {
    if (f.empty() || f.dims != 2 || f.channels() != 3) {
      throw std::invalid_argument("frame must be HxWx3 BGR");
    }
    cv::Mat u8;
    if (f.type() == CV_8UC3) {
      u8 = f;
    } else {
      cv::Mat f32;
      f.convertTo(f32, CV_32F);
      cv::min(f32, 255.0, f32);
      cv::max(f32, 0.0, f32);
      f32.convertTo(u8, CV_8UC3);
    }
    out.push_back(u8);
  }
  return out;
}

std::vector<Eigen::VectorXf> AVStreamInference::streamInference(
    const Eigen::Ref<const Eigen::VectorXf>& audio_chunk, const std::vector<cv::Mat>& video_chunk,
    bool is_start, bool is_end, int sampling_rate) {
  if (is_start || !started_) {
    reset();
  }

  const Eigen::VectorXf audio_in = normalizeAudioChunk(audio_chunk, sampling_rate);
  const std::vector<cv::Mat> video_in = normalizeVideoChunk(video_chunk);

  std::vector<cv::Mat> new_faces;
  new_faces.reserve(video_in.size());
  int scene_switch_hits = 0;
  for (const auto& fbgr : video_in) {
    auto [face_rgb, scene_switched] = tracker_.processBgr(fbgr, opt_.scene_switch_iou_thr);
    if (scene_switched) {
      scene_switch_hits++;
    }
    new_faces.push_back(face_rgb);
  }
  if (!new_faces.empty()) {
    resampler_.appendSrcFacesRgb255(new_faces);
  }
  if (opt_.clear_cache_on_scene_switch != 0 && scene_switch_hits > 0) {
    model_->clearStreamCache();
  }

  if (audio_in.size() > 0) {
    if (audio_buf_.size() == 0) {
      audio_buf_ = audio_in;
    } else {
      Eigen::VectorXf cat(audio_buf_.size() + audio_in.size());
      cat << audio_buf_, audio_in;
      audio_buf_ = std::move(cat);
    }
  }

  if (!use_stream_cache_ && produced_samples_ > 0 && !is_end &&
      audio_buf_.size() > max_history_samples_) {
    int n_drop_audio = static_cast<int>(audio_buf_.size()) - max_history_samples_;
    n_drop_audio = std::min(n_drop_audio, static_cast<int>(audio_buf_.size()));
    if (n_drop_audio > 0) {
      audio_buf_ = audio_buf_.tail(audio_buf_.size() - n_drop_audio);
      int n_drop_ref =
          static_cast<int>(std::lround(static_cast<float>(n_drop_audio) / audio_sr_ * ref_sr_));
      n_drop_ref = capRefTrimFrames(n_drop_ref, static_cast<int>(audio_buf_.size()), audio_sr_,
                                    ref_sr_, static_cast<int>(resampler_.tgtFrames().size()));
      if (n_drop_ref > 0) {
        resampler_.trimHead(n_drop_ref);
      }
      produced_samples_ = std::max(0, produced_samples_ - n_drop_audio);
    }
  }

  if (use_stream_cache_ && produced_samples_ > 0 && !is_end &&
      audio_buf_.size() > max_history_samples_) {
    const int excess = static_cast<int>(audio_buf_.size()) - max_history_samples_;
    const int units = excess / drop_unit_samples_;
    if (units > 0) {
      const int n_drop_audio = units * drop_unit_samples_;
      int n_drop_ref = units * drop_unit_ref_;
      n_drop_ref = capRefTrimFrames(n_drop_ref, static_cast<int>(audio_buf_.size()), audio_sr_,
                                    ref_sr_, static_cast<int>(resampler_.tgtFrames().size()));
      if (n_drop_audio < static_cast<int>(audio_buf_.size())) {
        audio_buf_ = audio_buf_.tail(audio_buf_.size() - n_drop_audio);
      } else {
        audio_buf_.resize(0);
      }
      resampler_.trimHead(n_drop_ref);
      produced_samples_ = std::max(0, produced_samples_ - n_drop_audio);
    }
  }

  std::vector<Eigen::VectorXf> outputs;
  const bool do_flush = is_end;
  if (!resampler_.tgtFrames().empty() && audio_buf_.size() > 0) {
    auto [wav_off, vid_off] =
        applyAvOffset(audio_buf_, resampler_.tgtFrames(), audio_sr_, ref_sr_, opt_.av_offset_ms);
    auto [wav_al, vid_list_al] = alignAudioVideoList(wav_off, vid_off, audio_sr_, ref_sr_);
    if (produced_samples_ > static_cast<int>(wav_al.size())) {
      produced_samples_ = static_cast<int>(wav_al.size());
    }
    const int available = static_cast<int>(wav_al.size()) - produced_samples_;
    if (available >= hop_samples_ || (do_flush && available > 0)) {
      HopRunResult hop = runNewHopsNonoverlap(wav_al, vid_list_al, *model_, hop_samples_, context_samples_,
                                              lookahead_samples_, audio_sr_, ref_sr_, produced_samples_,
                                              use_stream_cache_);
      produced_samples_ = std::min(hop.produced_samples, static_cast<int>(wav_al.size()));
      for (auto& seg : hop.segments) {
        outputs.push_back(std::move(seg));
      }
    }
    if (std::getenv("AV_TSE_DEBUG")) {
      std::cerr << "[AV_TSE_DEBUG] tgt=" << resampler_.tgtFrames().size()
                << " audio_buf=" << audio_buf_.size() << " wav_al=" << wav_al.size()
                << " vid=" << vid_list_al.size() << " produced=" << produced_samples_
                << " avail=" << available << " hop=" << hop_samples_
                << " cache=" << use_stream_cache_ << " segs=" << outputs.size() << " is_end=" << is_end
                << "\n";
    }
  }

  if (is_end) {
    reset();
  }
  return outputs;
}

}  // namespace av_tse
