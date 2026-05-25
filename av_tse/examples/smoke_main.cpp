#include <cstdlib>
#include <iostream>

#include <Eigen/Dense>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include "av_tse/av_mossformer_rknn.hpp"

int main(int argc, char** argv) {
  std::cout << "backend: RKNN (av_tse)\n";

  if (argc < 2) {
    std::cerr << "usage: " << argv[0] << " model/av_mossformer2.rknn\n";
    return 1;
  }

  const std::string rknn_path = argv[1];
  try {
    av_tse::AvMossformerRknn model(rknn_path);
    const int audio_len = model.fixedAudioLen();
    const int ref_frames = model.fixedRefFrames();
    const int image_size = model.imageSize();

    Eigen::VectorXf mixture = Eigen::VectorXf::Random(audio_len) * 1e-3f;
    constexpr float kMean = 0.506362f;
    constexpr float kStd = 0.272877f;
    cv::Mat neutral(image_size, image_size, CV_32FC3, cv::Scalar(128.f, 128.f, 128.f));
    neutral = (neutral - kMean) / kStd;
    std::vector<cv::Mat> refs(static_cast<size_t>(ref_frames), neutral);

    Eigen::VectorXf out = model.run(mixture, refs);
    std::cout << "smoke ok: mixture_len=" << audio_len << " ref_frames=" << ref_frames
              << " out_len=" << out.size() << "\n";
  } catch (const std::exception& e) {
    std::cerr << "smoke failed: " << e.what() << "\n";
    return 1;
  }
  return 0;
}
