#include "av_tse/av_align.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace av_tse {

int toVideoIndex(int sample_idx, int audio_sr, float ref_sr) {
  return static_cast<int>(std::floor(static_cast<float>(sample_idx) / static_cast<float>(audio_sr) *
                                     ref_sr));
}

std::pair<Eigen::VectorXf, std::vector<cv::Mat>> alignAudioVideoList(
    const Eigen::Ref<const Eigen::VectorXf>& audio, const std::vector<cv::Mat>& video_list,
    int audio_sr, float video_fps) {
  const int max_audio =
      static_cast<int>(std::floor(static_cast<float>(video_list.size()) / video_fps * audio_sr));
  if (max_audio <= 0) {
    throw std::runtime_error("video too short for alignment");
  }
  int min_audio = std::min(static_cast<int>(audio.size()), max_audio);
  Eigen::VectorXf audio_cut = audio.head(min_audio);

  // Compute how many video frames correspond to the trimmed audio.
  int keep_video = static_cast<int>(
      std::ceil(static_cast<float>(audio_cut.size()) / static_cast<float>(audio_sr) * video_fps));
  keep_video = std::max(1, std::min(keep_video, static_cast<int>(video_list.size())));
  std::vector<cv::Mat> video_cut(video_list.begin(), video_list.begin() + keep_video);
  // Keep the full audio (min_audio) — do not re-truncate from keep_video back to audio,
  // because the floor round-trip (audio→frames→audio) systematically drops samples.
  return {audio_cut, video_cut};
}

std::pair<Eigen::VectorXf, std::vector<cv::Mat>> applyAvOffset(
    const Eigen::Ref<const Eigen::VectorXf>& audio, const std::vector<cv::Mat>& video_list,
    int audio_sr, float video_fps, float av_offset_ms) {
  if (std::abs(av_offset_ms) < 1e-9f || audio.size() == 0 || video_list.empty()) {
    return {audio, video_list};
  }
  if (av_offset_ms > 0.f) {
    int off_samples = static_cast<int>(std::lround(av_offset_ms * audio_sr / 1000.f));
    off_samples = std::clamp(off_samples, 0, static_cast<int>(audio.size()));
    return {audio.tail(audio.size() - off_samples), video_list};
  }
  int off_frames = static_cast<int>(std::lround(std::abs(av_offset_ms) * video_fps / 1000.f));
  off_frames = std::clamp(off_frames, 0, static_cast<int>(video_list.size()));
  std::vector<cv::Mat> v(video_list.begin() + off_frames, video_list.end());
  return {audio, v};
}

static Eigen::VectorXf padOrTrim(const Eigen::Ref<const Eigen::VectorXf>& seg, int target_len) {
  if (seg.size() == target_len) {
    return seg;
  }
  if (seg.size() < target_len) {
    Eigen::VectorXf out(target_len);
    out.setZero();
    out.head(seg.size()) = seg;
    return out;
  }
  return seg.head(target_len);
}

HopRunResult runNewHopsNonoverlap(const Eigen::Ref<const Eigen::VectorXf>& wav_al,
                                  const std::vector<cv::Mat>& vid_norm_list,
                                  AvMossformerModel& model, int hop_samples, int context_samples,
                                  int lookahead_samples, int audio_sr, float ref_sr,
                                  int produced_samples, bool use_stream_cache) {
  HopRunResult result;
  result.produced_samples = produced_samples;
  const int total_samples = static_cast<int>(wav_al.size());
  if (total_samples <= 0 || vid_norm_list.empty()) {
    return result;
  }

  if (use_stream_cache) {
    int p0 = std::min(produced_samples, total_samples);
    const int end_emit = std::max(p0, total_samples - std::max(0, lookahead_samples));
    if (end_emit <= p0) {
      return result;
    }
    Eigen::VectorXf y_full = model.run(wav_al, vid_norm_list);
    int p = p0;
    while (p < end_emit) {
      const int cur_end = std::min(p + hop_samples, end_emit);
      const int target_len = cur_end - p;
      const int avail = std::max(0, static_cast<int>(y_full.size()) - p);
      const int take = std::min(target_len, avail);
      Eigen::VectorXf seg = (take > 0) ? y_full.segment(p, take) : Eigen::VectorXf(0);
      result.segments.push_back(padOrTrim(seg, target_len));
      p = cur_end;
    }
    result.produced_samples = p;
    return result;
  }

  int p = produced_samples;
  while (p < total_samples) {
    const int cur_start = p;
    const int cur_end = std::min(cur_start + hop_samples, total_samples);
    const int win_start = std::max(0, cur_start - context_samples);
    const int win_end = std::min(total_samples, cur_end + lookahead_samples);

    int v_start = toVideoIndex(win_start, audio_sr, ref_sr);
    int v_end = static_cast<int>(std::ceil(static_cast<float>(win_end) / audio_sr * ref_sr));
    v_start = std::clamp(v_start, 0, static_cast<int>(vid_norm_list.size()) - 1);
    v_end = std::clamp(std::max(v_start + 1, v_end), v_start + 1,
                       static_cast<int>(vid_norm_list.size()));

    if (win_end <= win_start || v_end <= v_start) {
      break;
    }

    Eigen::VectorXf a_in = wav_al.segment(win_start, win_end - win_start);
    std::vector<cv::Mat> r_in(vid_norm_list.begin() + v_start, vid_norm_list.begin() + v_end);
    Eigen::VectorXf y_win = model.run(a_in, r_in);

    const int seg_local_start = std::max(0, cur_start - win_start);
    const int seg_local_end =
        std::max(seg_local_start, std::min(cur_end - win_start, static_cast<int>(y_win.size())));
    Eigen::VectorXf seg = y_win.segment(seg_local_start, seg_local_end - seg_local_start);
    const int target_len = cur_end - cur_start;
    result.segments.push_back(padOrTrim(seg, target_len));
    p = cur_end;
  }
  result.produced_samples = p;
  return result;
}

}  // namespace av_tse
