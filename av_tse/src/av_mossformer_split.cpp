#include "av_tse/av_mossformer_split.hpp"

#include <cmath>
#include <stdexcept>

namespace av_tse {

namespace {

constexpr float kGrayR = 0.2989f;
constexpr float kGrayG = 0.5870f;
constexpr float kGrayB = 0.1140f;

}  // namespace

cv::Mat AvMossformerSplit::rgbToGray(const cv::Mat& rgb) {
  cv::Mat gray;
  if (rgb.channels() == 3) {
    cv::Mat f32;
    if (rgb.type() == CV_32FC3) {
      f32 = rgb;
    } else {
      rgb.convertTo(f32, CV_32F, 1.0 / 255.0);
    }
    gray.create(f32.rows, f32.cols, CV_32FC1);
    for (int y = 0; y < f32.rows; ++y) {
      const cv::Vec3f* in_row = f32.ptr<cv::Vec3f>(y);
      float* out_row = gray.ptr<float>(y);
      for (int x = 0; x < f32.cols; ++x) {
        out_row[x] = kGrayR * in_row[x][0] + kGrayG * in_row[x][1] + kGrayB * in_row[x][2];
      }
    }
    return gray;
  }
  if (rgb.channels() == 1) {
    if (rgb.type() == CV_32FC1) {
      return rgb.clone();
    }
    cv::Mat out;
    rgb.convertTo(out, CV_32F, 1.0 / 255.0);
    return out;
  }
  throw std::runtime_error("ref frame must be 1 or 3 channels");
}

AvMossformerSplit::AvMossformerSplit(const std::string& ref_onnx_path, const std::string& rknn_sep_path,
                                     int audio_len, int ref_frames, int image_size, int onnx_num_threads)
    : ref_encoder_(std::make_unique<AvMossformerRefOnnx>(ref_onnx_path, onnx_num_threads)),
      sep_rknn_(std::make_unique<AvMossformerRknn>(rknn_sep_path, audio_len, ref_frames, image_size)),
      image_size_(image_size) {}

Eigen::VectorXf AvMossformerSplit::run(const Eigen::Ref<const Eigen::VectorXf>& mixture,
                                       const std::vector<cv::Mat>& ref_frames) {
  std::vector<cv::Mat> gray_frames;
  gray_frames.reserve(ref_frames.size());
  for (const auto& f : ref_frames) {
    gray_frames.push_back(rgbToGray(f));
  }
  const std::vector<float> ref_feat =
      ref_encoder_->runGrayFrames(gray_frames, image_size_);
  return sep_rknn_->runWithRefFeat(mixture, ref_feat);
}

}  // namespace av_tse
