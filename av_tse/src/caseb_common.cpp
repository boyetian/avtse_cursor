#include "av_tse/caseb_runner.hpp"

#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dirent.h>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <opencv2/imgcodecs.hpp>
#if defined(AV_TSE_HAVE_VIDEOIO)
#include <opencv2/videoio.hpp>
#endif

#include "av_tse/stream_inference_sdk.hpp"

namespace av_tse {
namespace {

std::string pathJoin(const std::string& a, const std::string& b) {
  if (a.empty()) {
    return b;
  }
  if (b.empty()) {
    return a;
  }
  if (a.back() == '/') {
    return a + b;
  }
  return a + "/" + b;
}

bool pathExists(const std::string& p) {
  struct stat st {};
  return stat(p.c_str(), &st) == 0;
}

bool isDirectory(const std::string& p) {
  struct stat st {};
  if (stat(p.c_str(), &st) != 0) {
    return false;
  }
  return S_ISDIR(st.st_mode);
}

bool createDirectories(const std::string& path) {
  if (path.empty()) {
    return false;
  }
  std::string cur;
  for (size_t i = 0; i < path.size(); ++i) {
    const char c = path[i];
    cur.push_back(c);
    if (c == '/' && cur.size() > 1) {
      mkdir(cur.c_str(), 0755);
    }
  }
  return mkdir(path.c_str(), 0755) == 0 || pathExists(path);
}

bool removeAllInDir(const std::string& dir) {
  DIR* d = opendir(dir.c_str());
  if (!d) {
    return false;
  }
  struct dirent* ent;
  while ((ent = readdir(d)) != nullptr) {
    if (std::strcmp(ent->d_name, ".") == 0 || std::strcmp(ent->d_name, "..") == 0) {
      continue;
    }
    const std::string child = pathJoin(dir, ent->d_name);
    struct stat st {};
    if (stat(child.c_str(), &st) == 0 && S_ISDIR(st.st_mode)) {
      removeAllInDir(child);
      rmdir(child.c_str());
    } else {
      std::remove(child.c_str());
    }
  }
  closedir(d);
  return true;
}

std::string tempFrameDir() {
#ifdef __ANDROID__
  return "/data/local/tmp/av_tse_caseb_frames";
#else
  const char* tmp = std::getenv("TMPDIR");
  if (!tmp || tmp[0] == '\0') {
    tmp = "/tmp";
  }
  return pathJoin(tmp, "av_tse_caseb_frames");
#endif
}

struct WavPcm16 {
  int sample_rate = 0;
  int channels = 0;
  std::vector<int16_t> interleaved;
};

bool readRiffChunk(std::ifstream& f, char id[4], uint32_t& size) {
  return static_cast<bool>(f.read(id, 4) && f.read(reinterpret_cast<char*>(&size), 4));
}

bool loadWavPcm16(const std::string& path, WavPcm16& out) {
  std::ifstream f(path, std::ios::binary);
  if (!f) {
    return false;
  }
  char riff[4], wave[4];
  uint32_t riff_size = 0;
  if (!f.read(riff, 4) || std::memcmp(riff, "RIFF", 4) != 0) {
    return false;
  }
  if (!f.read(reinterpret_cast<char*>(&riff_size), 4)) {
    return false;
  }
  if (!f.read(wave, 4) || std::memcmp(wave, "WAVE", 4) != 0) {
    return false;
  }

  bool got_fmt = false;
  bool got_data = false;
  char id[4];
  uint32_t chunk_size = 0;
  while (readRiffChunk(f, id, chunk_size)) {
    if (std::memcmp(id, "fmt ", 4) == 0) {
      if (chunk_size < 16) {
        return false;
      }
      uint16_t audio_format = 0;
      uint16_t num_channels = 0;
      uint32_t sample_rate = 0;
      uint32_t byte_rate = 0;
      uint16_t block_align = 0;
      uint16_t bits_per_sample = 0;
      f.read(reinterpret_cast<char*>(&audio_format), 2);
      f.read(reinterpret_cast<char*>(&num_channels), 2);
      f.read(reinterpret_cast<char*>(&sample_rate), 4);
      f.read(reinterpret_cast<char*>(&byte_rate), 4);
      f.read(reinterpret_cast<char*>(&block_align), 2);
      f.read(reinterpret_cast<char*>(&bits_per_sample), 2);
      if (chunk_size > 16) {
        f.seekg(chunk_size - 16, std::ios::cur);
      }
      if (audio_format != 1 || bits_per_sample != 16) {
        return false;
      }
      out.sample_rate = static_cast<int>(sample_rate);
      out.channels = static_cast<int>(num_channels);
      got_fmt = true;
    } else if (std::memcmp(id, "data", 4) == 0) {
      if (!got_fmt) {
        return false;
      }
      const size_t n_bytes = chunk_size;
      out.interleaved.resize(n_bytes / 2);
      if (!f.read(reinterpret_cast<char*>(out.interleaved.data()),
                 static_cast<std::streamsize>(n_bytes))) {
        return false;
      }
      got_data = true;
      break;
    } else {
      f.seekg(chunk_size + (chunk_size & 1), std::ios::cur);
    }
  }
  return got_fmt && got_data && out.channels > 0 &&
         out.interleaved.size() % static_cast<size_t>(out.channels) == 0;
}

Eigen::MatrixXf wavToMatrix(const WavPcm16& w) {
  const int frames = static_cast<int>(w.interleaved.size() / static_cast<size_t>(w.channels));
  Eigen::MatrixXf m(w.channels, frames);
  for (int t = 0; t < frames; ++t) {
    for (int c = 0; c < w.channels; ++c) {
      const int16_t s = w.interleaved[static_cast<size_t>(t * w.channels + c)];
      m(c, t) = static_cast<float>(s) / 32768.0f;
    }
  }
  return m;
}

bool writeWavMonoPcm16(const std::string& path, const std::vector<int16_t>& mono, int sample_rate) {
  const size_t slash = path.find_last_of('/');
  if (slash != std::string::npos) {
    createDirectories(path.substr(0, slash));
  }
  std::ofstream f(path, std::ios::binary);
  if (!f) {
    return false;
  }
  const uint32_t data_bytes = static_cast<uint32_t>(mono.size() * sizeof(int16_t));
  const uint32_t riff_chunk_size = 36 + data_bytes;
  const uint16_t audio_format = 1;
  const uint16_t num_channels = 1;
  const uint16_t bits_per_sample = 16;
  const uint16_t block_align = num_channels * (bits_per_sample / 8);
  const uint32_t byte_rate = static_cast<uint32_t>(sample_rate) * block_align;

  f.write("RIFF", 4);
  f.write(reinterpret_cast<const char*>(&riff_chunk_size), 4);
  f.write("WAVE", 4);
  f.write("fmt ", 4);
  const uint32_t sub1 = 16;
  f.write(reinterpret_cast<const char*>(&sub1), 4);
  f.write(reinterpret_cast<const char*>(&audio_format), 2);
  f.write(reinterpret_cast<const char*>(&num_channels), 2);
  f.write(reinterpret_cast<const char*>(&sample_rate), 4);
  f.write(reinterpret_cast<const char*>(&byte_rate), 4);
  f.write(reinterpret_cast<const char*>(&block_align), 2);
  f.write(reinterpret_cast<const char*>(&bits_per_sample), 2);
  f.write("data", 4);
  f.write(reinterpret_cast<const char*>(&data_bytes), 4);
  f.write(reinterpret_cast<const char*>(mono.data()), static_cast<std::streamsize>(data_bytes));
  return static_cast<bool>(f);
}

std::string resolveFfmpegBinary() {
  if (const char* env = std::getenv("FFMPEG_PATH")) {
    if (env[0] != '\0') {
      return env;
    }
  }
  const char* candidates[] = {"ffmpeg", "/system/bin/ffmpeg", "/vendor/bin/ffmpeg"};
  for (const char* bin : candidates) {
    const std::string cmd = std::string(bin) + " -version >/dev/null 2>&1";
    if (std::system(cmd.c_str()) == 0) {
      return bin;
    }
  }
  return "ffmpeg";
}

float probeVideoFps(const std::string& mp4_path) {
  const std::string ffmpeg = resolveFfmpegBinary();
  const std::string cmd = "\"" + ffmpeg + "\" -v error -select_streams v:0 -show_entries stream=r_frame_rate "
                          "-of default=nw=1:nk=1 \"" +
                          mp4_path + "\" 2>/dev/null";
  FILE* pipe = popen(cmd.c_str(), "r");
  if (!pipe) {
    return 30.f;
  }
  char buf[64] = {};
  if (fgets(buf, sizeof(buf), pipe) == nullptr) {
    pclose(pipe);
    return 30.f;
  }
  pclose(pipe);
  const std::string rate_str(buf);
  const auto slash = rate_str.find('/');
  try {
    if (slash != std::string::npos) {
      const float num = std::stof(rate_str.substr(0, slash));
      const float den = std::stof(rate_str.substr(slash + 1));
      if (den > 1e-6f) {
        return num / den;
      }
    }
    return std::stof(rate_str);
  } catch (...) {
    return 30.f;
  }
}

float resolveFpsOverride(float probed) {
  if (const char* env = std::getenv("AV_TSE_CASEB_FPS")) {
    const float fps = std::strtof(env, nullptr);
    if (std::isfinite(fps) && fps > 1e-3f) {
      return fps;
    }
  }
  if (std::isfinite(probed) && probed > 1e-3f) {
    return probed;
  }
  return 30.f;
}

std::vector<std::string> listImagePaths(const std::string& frames_dir) {
  std::vector<std::string> frame_paths;
  DIR* d = opendir(frames_dir.c_str());
  if (!d) {
    return frame_paths;
  }
  struct dirent* ent;
  while ((ent = readdir(d)) != nullptr) {
    const std::string name = ent->d_name;
    if (name.size() >= 4) {
      const std::string ext = name.substr(name.size() - 4);
      if (ext == ".jpg" || ext == ".png" || (name.size() >= 5 && name.substr(name.size() - 5) == ".jpeg")) {
        frame_paths.push_back(pathJoin(frames_dir, name));
      }
    }
  }
  closedir(d);
  std::sort(frame_paths.begin(), frame_paths.end());
  return frame_paths;
}

std::pair<std::vector<cv::Mat>, float> loadVideoFramesFromDir(const std::string& frames_dir) {
  const auto frame_paths = listImagePaths(frames_dir);
  std::vector<cv::Mat> frames;
  frames.reserve(frame_paths.size());
  for (const auto& p : frame_paths) {
    cv::Mat img = cv::imread(p, cv::IMREAD_COLOR);
    if (!img.empty()) {
      frames.push_back(std::move(img));
    }
  }
  if (frames.empty()) {
    throw std::runtime_error("no frames in directory: " + frames_dir);
  }
  return {frames, resolveFpsOverride(30.f)};
}

std::pair<std::vector<cv::Mat>, float> loadVideoFramesFfmpegCli(const std::string& mp4_path) {
  const std::string tmp = tempFrameDir();
  removeAllInDir(tmp);
  createDirectories(tmp);

  const std::string ffmpeg = resolveFfmpegBinary();
  const std::string out_pattern = pathJoin(tmp, "frame_%06d.jpg");
  const std::string cmd =
      "\"" + ffmpeg + "\" -loglevel error -y -i \"" + mp4_path + "\" \"" + out_pattern + "\"";
  if (std::system(cmd.c_str()) != 0) {
    throw std::runtime_error("ffmpeg decode failed for: " + mp4_path);
  }

  auto result = loadVideoFramesFromDir(tmp);
  float fps = probeVideoFps(mp4_path);
  result.second = resolveFpsOverride(fps);
  removeAllInDir(tmp);
  return result;
}

#if defined(AV_TSE_HAVE_VIDEOIO)
std::pair<std::vector<cv::Mat>, float> loadVideoFramesCapture(const std::string& mp4_path) {
  cv::VideoCapture cap(mp4_path);
  if (!cap.isOpened()) {
    throw std::runtime_error("VideoCapture failed: " + mp4_path);
  }
  cv::Mat probe;
  if (!cap.read(probe) || probe.empty()) {
    throw std::runtime_error("VideoCapture empty first frame: " + mp4_path);
  }
  cap.set(cv::CAP_PROP_POS_FRAMES, 0);
  float fps = static_cast<float>(cap.get(cv::CAP_PROP_FPS));
  fps = resolveFpsOverride(fps);
  std::vector<cv::Mat> frames;
  frames.push_back(std::move(probe));
  cv::Mat frame;
  while (cap.read(frame)) {
    frames.push_back(frame.clone());
  }
  return {frames, fps};
}
#endif

size_t countFrameImages(const std::string& frames_dir) {
  return listImagePaths(frames_dir).size();
}

std::pair<std::vector<cv::Mat>, float> loadVideoFrames(const std::string& mp4_path,
                                                       const std::string& frames_dir,
                                                       CaseBLayout layout) {
  if (layout == CaseBLayout::AndroidFlat) {
    if (frames_dir.empty() || !isDirectory(frames_dir)) {
      throw std::runtime_error("missing or invalid video frames dir: " + frames_dir);
    }
    if (countFrameImages(frames_dir) == 0) {
      throw std::runtime_error("no frame images in directory: " + frames_dir);
    }
    std::cout << "Loading pre-extracted frames from " << frames_dir << "\n";
    return loadVideoFramesFromDir(frames_dir);
  }

  if (!frames_dir.empty() && isDirectory(frames_dir) && countFrameImages(frames_dir) > 0) {
    std::cout << "Loading pre-extracted frames from " << frames_dir << "\n";
    return loadVideoFramesFromDir(frames_dir);
  }
#if defined(AV_TSE_HAVE_VIDEOIO)
  if (pathExists(mp4_path)) {
    cv::VideoCapture cap(mp4_path);
    if (cap.isOpened()) {
      cv::Mat probe;
      if (cap.read(probe) && !probe.empty()) {
        std::cout << "Loading video via OpenCV VideoCapture\n";
        cap.set(cv::CAP_PROP_POS_FRAMES, 0);
        return loadVideoFramesCapture(mp4_path);
      }
    }
  }
#endif
  if (pathExists(mp4_path)) {
    std::cout << "Decoding mp4 via ffmpeg CLI\n";
    return loadVideoFramesFfmpegCli(mp4_path);
  }
  throw std::runtime_error("no video: missing mp4 and frames directory");
}

struct CaseBResult {
  std::vector<Eigen::VectorXf> outputs;
  double sum_process_s = 0.0;
  double audio_dur_s = 0.0;
};

CaseBResult runCaseBFullNumpy(StreamInferenceSDK& streamer, const Eigen::MatrixXf& wav_np,
                              const std::vector<cv::Mat>& frames_np, float chunk_ms, int sr,
                              float fps) {
  const int audio_step =
      std::max(1, static_cast<int>(std::lround(static_cast<float>(sr) * (chunk_ms / 1000.f))));
  Eigen::MatrixXf wav_arr = wav_np;
  if (wav_arr.rows() != 1 && wav_arr.cols() == 1) {
    wav_arr = wav_arr.transpose().eval();
  }

  std::vector<Eigen::VectorXf> audio_chunk_list;
  for (int i = 0; i < wav_arr.cols(); i += audio_step) {
    const int len = std::min(audio_step, static_cast<int>(wav_arr.cols()) - i);
    if (wav_arr.rows() == 1) {
      audio_chunk_list.push_back(wav_arr.row(0).segment(i, len));
    } else {
      audio_chunk_list.push_back(wav_arr.colwise().mean().segment(i, len));
    }
  }

  std::vector<std::vector<cv::Mat>> video_chunk_list;
  int v_pos = 0;
  cv::Mat last_frame = frames_np.back();
  for (const auto& a_chunk : audio_chunk_list) {
    const float dur_s = static_cast<float>(a_chunk.size()) / static_cast<float>(std::max(1, sr));
    const int n_frames = std::max(1, static_cast<int>(std::lround(fps * dur_s)));
    std::vector<cv::Mat> one;
    one.reserve(static_cast<size_t>(n_frames));
    for (int k = 0; k < n_frames; ++k) {
      if (v_pos < static_cast<int>(frames_np.size())) {
        last_frame = frames_np[static_cast<size_t>(v_pos)];
        one.push_back(last_frame);
        v_pos++;
      } else {
        one.push_back(last_frame);
      }
    }
    video_chunk_list.push_back(std::move(one));
  }

  CaseBResult result;
  const int total = static_cast<int>(std::min(audio_chunk_list.size(), video_chunk_list.size()));
  for (int i = 0; i < total; ++i) {
    const auto t0 = std::chrono::steady_clock::now();
    auto outs = streamer.processAvStream(audio_chunk_list[static_cast<size_t>(i)],
                                         video_chunk_list[static_cast<size_t>(i)], i == 0,
                                         i == total - 1, sr, fps);
    result.sum_process_s += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    const std::size_t n_out = outs.size();
    for (auto& o : outs) {
      result.outputs.push_back(std::move(o));
    }
    std::cout << "[Case B] 进度: " << (i + 1) << "/" << total << "，本次输出 " << n_out
              << " 段\n";
  }
  result.audio_dur_s = static_cast<double>(wav_arr.cols()) / static_cast<double>(std::max(1, sr));
  return result;
}

void applyModelPath(StreamInferenceSDKOptions& opt, const std::string& model_path) {
#if defined(AV_TSE_USE_RKNN) && AV_TSE_USE_RKNN
  opt.rknn_path = model_path;
#else
  opt.onnx_path = model_path;
#endif
}

std::string defaultModelPath(const std::string& root, CaseBLayout layout) {
#if defined(AV_TSE_USE_RKNN) && AV_TSE_USE_RKNN
  const char* ext = ".rknn";
#else
  const char* ext = ".onnx";
#endif
  if (layout == CaseBLayout::AndroidFlat) {
    return pathJoin(pathJoin(root, "model"), std::string("av_mossformer2") + ext);
  }
  return pathJoin(pathJoin(pathJoin(root, "checkpoints"), "AV_Mossformer"),
                  std::string("av_mossformer2") + ext);
}

}  // namespace

CaseBPaths resolveCaseBPaths(const std::string& data_root, CaseBLayout layout) {
  CaseBPaths p;
  p.layout = layout;
  p.data_root = data_root;
  if (layout == CaseBLayout::AndroidFlat) {
    p.audio_wav = pathJoin(data_root, "audio/test03.wav");
    p.video_mp4 = pathJoin(data_root, "video/test03.mp4");
    p.video_frames_dir = pathJoin(data_root, "video/test03_frames");
    p.config_yaml = pathJoin(data_root, "model/config.yaml");
    p.model_path = defaultModelPath(data_root, layout);
    p.output_wav = pathJoin(pathJoin(data_root, "out"), "test03_out.wav");
  } else {
    p.audio_wav = pathJoin(data_root, "测试用例/音频/test03.wav");
    p.video_mp4 = pathJoin(data_root, "测试用例/视频/test03.mp4");
    p.video_frames_dir = pathJoin(data_root, "测试用例/视频/test03_frames");
    p.config_yaml = pathJoin(pathJoin(pathJoin(data_root, "checkpoints"), "AV_Mossformer"), "config.yaml");
    p.model_path = defaultModelPath(data_root, layout);
    p.output_wav = pathJoin(pathJoin(data_root, "out"), "test03_out.wav");
  }
  return p;
}

int runCaseB(const CaseBPaths& paths) {
  if (!pathExists(paths.audio_wav)) {
    std::cerr << "missing audio: " << paths.audio_wav << "\n";
    return 2;
  }
  if (!pathExists(paths.config_yaml)) {
    std::cerr << "missing config: " << paths.config_yaml << "\n";
    return 2;
  }
  if (!pathExists(paths.model_path)) {
    std::cerr << "missing model: " << paths.model_path << "\n";
    return 2;
  }
  if (paths.layout == CaseBLayout::AndroidFlat) {
    if (paths.video_frames_dir.empty() || !isDirectory(paths.video_frames_dir)) {
      std::cerr << "missing video frames dir: " << paths.video_frames_dir << "\n";
      return 2;
    }
    if (countFrameImages(paths.video_frames_dir) == 0) {
      std::cerr << "no frame images in: " << paths.video_frames_dir << "\n";
      return 2;
    }
  } else {
    const bool has_mp4 = pathExists(paths.video_mp4);
    const bool has_frames =
        !paths.video_frames_dir.empty() && isDirectory(paths.video_frames_dir) &&
        countFrameImages(paths.video_frames_dir) > 0;
    if (!has_mp4 && !has_frames) {
      std::cerr << "missing video mp4 (" << paths.video_mp4 << ") and frames dir ("
                << paths.video_frames_dir << ")\n";
      return 2;
    }
  }

  WavPcm16 wav_file;
  if (!loadWavPcm16(paths.audio_wav, wav_file)) {
    std::cerr << "failed to read wav: " << paths.audio_wav << "\n";
    return 3;
  }
  Eigen::MatrixXf wav = wavToMatrix(wav_file);
  const int sr = wav_file.sample_rate;

  std::vector<cv::Mat> frames;
  float fps = 30.f;
  try {
    std::tie(frames, fps) =
        loadVideoFrames(paths.video_mp4, paths.video_frames_dir, paths.layout);
  } catch (const std::exception& e) {
    std::cerr << "video load failed: " << e.what() << "\n";
    return 4;
  }

  StreamInferenceSDKOptions opt;
  opt.config_yaml = paths.config_yaml;
  if (const char* env = std::getenv("AV_TSE_TEST_MODEL_PATH")) {
    if (env[0] != '\0') {
      applyModelPath(opt, env);
    } else {
      applyModelPath(opt, paths.model_path);
    }
  } else if (const char* env = std::getenv("AV_TSE_CASEB_MODEL_PATH")) {
    if (env[0] != '\0') {
      applyModelPath(opt, env);
    } else {
      applyModelPath(opt, paths.model_path);
    }
  } else {
    applyModelPath(opt, paths.model_path);
  }
  opt.infer_chunk_ms = 500.f;
  opt.core_infer_chunk_ms = 0.f;
  opt.context_ms = 100.f;
  opt.max_history_ms = 600.f;
  opt.use_stream_cache = 1;

  std::unique_ptr<StreamInferenceSDK> streamer;
  try {
    streamer = std::make_unique<StreamInferenceSDK>(opt);
  } catch (const std::exception& e) {
    std::cerr << "StreamInferenceSDK init failed: " << e.what() << "\n";
    return 5;
  }

  auto case_b = runCaseBFullNumpy(*streamer, wav, frames, 100.f, sr, fps);
  streamer->close();

  const double rtf =
      (case_b.audio_dur_s > 1e-9) ? (case_b.sum_process_s / case_b.audio_dur_s) : 0.0;
  std::cout << "RTF: " << rtf << "  (sum_sdk=" << case_b.sum_process_s
            << "s, audio_dur=" << case_b.audio_dur_s << "s)\n";

  if (case_b.outputs.empty()) {
    std::cerr << "Case B produced no audio output\n";
    return 6;
  }

  Eigen::Index out_samples = 0;
  for (const auto& v : case_b.outputs) {
    out_samples += v.size();
  }
  Eigen::VectorXf out_audio(out_samples);
  Eigen::Index off = 0;
  for (const auto& v : case_b.outputs) {
    out_audio.segment(off, v.size()) = v;
    off += v.size();
  }

  Eigen::VectorXf save_wav = out_audio;
  const float mx = save_wav.cwiseAbs().maxCoeff();
  if (mx <= 1.f) {
    save_wav *= 32767.f;
  }
  std::vector<int16_t> pcm(static_cast<size_t>(save_wav.size()));
  for (Eigen::Index i = 0; i < save_wav.size(); ++i) {
    float x = std::max(-32768.f, std::min(32767.f, save_wav[i]));
    pcm[static_cast<size_t>(i)] = static_cast<int16_t>(std::lround(x));
  }

  if (!writeWavMonoPcm16(paths.output_wav, pcm, sr)) {
    std::cerr << "failed to write " << paths.output_wav << "\n";
    return 7;
  }
  std::cout << "已保存: " << paths.output_wav << "\n";

  const double out_dur_s = static_cast<double>(out_samples) / static_cast<double>(std::max(1, sr));
  if (std::abs(out_dur_s - case_b.audio_dur_s) > 0.10) {
    std::cerr << "duration mismatch: out=" << out_dur_s << "s expected~=" << case_b.audio_dur_s
              << "s\n";
    return 8;
  }
  return 0;
}

}  // namespace av_tse
