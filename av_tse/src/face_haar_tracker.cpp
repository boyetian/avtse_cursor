#include "av_tse/face_haar_tracker.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include <opencv2/imgproc.hpp>

namespace av_tse {

FaceHaarStreamTracker::FaceHaarStreamTracker(const FaceHaarTrackerOptions& opt) : opt_(opt) {
  std::string cascade =
      cv::samples::findFile("haarcascades/haarcascade_frontalface_default.xml", false, false);
#ifdef AV_TSE_HAAR_CASCADE_PATH
  if (cascade.empty()) {
    cascade = AV_TSE_HAAR_CASCADE_PATH;
  }
#endif
  if (cascade.empty()) {
    cascade = "/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml";
  }
  if (!detector_.load(cascade)) {
    throw std::runtime_error("failed to load OpenCV haarcascade: " + cascade);
  }
}

float FaceHaarStreamTracker::boxIouXyxy(const float* a, const float* b) {
  const float ix1 = std::max(a[0], b[0]);
  const float iy1 = std::max(a[1], b[1]);
  const float ix2 = std::min(a[2], b[2]);
  const float iy2 = std::min(a[3], b[3]);
  const float iw = std::max(0.f, ix2 - ix1);
  const float ih = std::max(0.f, iy2 - iy1);
  const float inter = iw * ih;
  const float area_a = std::max(0.f, a[2] - a[0]) * std::max(0.f, a[3] - a[1]);
  const float area_b = std::max(0.f, b[2] - b[0]) * std::max(0.f, b[3] - b[1]);
  const float den = area_a + area_b - inter;
  return (den > 1e-9f) ? (inter / den) : 0.f;
}

std::pair<cv::Mat, bool> FaceHaarStreamTracker::processBgr(const cv::Mat& frame_bgr,
                                                          float scene_switch_iou_thr) {
  const int h = frame_bgr.rows;
  const int w = frame_bgr.cols;
  const bool run_det = (frame_idx_ % std::max(1, opt_.detect_every_n) == 0) || !last_box_;
  bool scene_switched = false;

  if (run_det) {
    cv::Mat gray;
    cv::cvtColor(frame_bgr, gray, cv::COLOR_BGR2GRAY);
    cv::Mat det_gray = gray;
    float sx = 1.f;
    float sy = 1.f;
    if (opt_.detect_max_side > 0) {
      const int ms = std::max(h, w);
      if (ms > opt_.detect_max_side) {
        const float scale = static_cast<float>(opt_.detect_max_side) / static_cast<float>(ms);
        const int sw = std::max(1, static_cast<int>(std::lround(w * scale)));
        const int sh = std::max(1, static_cast<int>(std::lround(h * scale)));
        cv::resize(gray, det_gray, cv::Size(sw, sh), 0, 0, cv::INTER_AREA);
        sx = static_cast<float>(w) / static_cast<float>(sw);
        sy = static_cast<float>(h) / static_cast<float>(sh);
      }
    }
    const int min_sz = std::max(20, static_cast<int>(30.f * std::min(sx, sy)));
    std::vector<cv::Rect> faces;
    detector_.detectMultiScale(det_gray, faces, opt_.haar_scale_factor, opt_.haar_min_neighbors, 0,
                               cv::Size(min_sz, min_sz));

    if (!faces.empty()) {
      int best = 0;
      int best_area = faces[0].width * faces[0].height;
      for (size_t i = 1; i < faces.size(); ++i) {
        const int area = faces[i].width * faces[i].height;
        if (area > best_area) {
          best_area = area;
          best = static_cast<int>(i);
        }
      }
      int x = static_cast<int>(std::lround(faces[best].x * sx));
      int y = static_cast<int>(std::lround(faces[best].y * sy));
      int fw = static_cast<int>(std::lround(faces[best].width * sx));
      int fh = static_cast<int>(std::lround(faces[best].height * sy));
      const float cx = x + fw / 2.f;
      const float cy = y + fh / 2.f;
      const float s = static_cast<float>(std::max(fw, fh)) * opt_.face_scale;
      std::array<float, 4> new_box{cx - s / 2.f, cy - s / 2.f, cx + s / 2.f, cy + s / 2.f};

      if (last_detected_box_) {
        if (boxIouXyxy(last_detected_box_->data(), new_box.data()) < scene_switch_iou_thr) {
          scene_switched = true;
        }
      }
      last_detected_box_ = new_box;

      if (!last_box_ || opt_.box_smooth_alpha >= 0.999f) {
        last_box_ = new_box;
      } else {
        for (int i = 0; i < 4; ++i) {
          (*last_box_)[i] =
              opt_.box_smooth_alpha * (*last_box_)[i] + (1.f - opt_.box_smooth_alpha) * new_box[i];
        }
      }
    }
  }

  int x1, y1, x2, y2;
  if (!last_box_) {
    const int cs = std::min(h, w);
    x1 = (w - cs) / 2;
    y1 = (h - cs) / 2;
    x2 = x1 + cs;
    y2 = y1 + cs;
  } else {
    x1 = static_cast<int>(std::lround(std::max(0.f, std::min(static_cast<float>(w - 1), (*last_box_)[0]))));
    y1 = static_cast<int>(std::lround(std::max(0.f, std::min(static_cast<float>(h - 1), (*last_box_)[1]))));
    x2 = static_cast<int>(std::lround(std::max(static_cast<float>(x1 + 1), std::min(static_cast<float>(w), (*last_box_)[2]))));
    y2 = static_cast<int>(std::lround(std::max(static_cast<float>(y1 + 1), std::min(static_cast<float>(h), (*last_box_)[3]))));
  }

  cv::Mat crop = frame_bgr(cv::Rect(x1, y1, std::max(1, x2 - x1), std::max(1, y2 - y1))).clone();
  if (crop.empty()) {
    const int cs = std::min(h, w);
    x1 = (w - cs) / 2;
    y1 = (h - cs) / 2;
    crop = frame_bgr(cv::Rect(x1, y1, cs, cs)).clone();
  }
  cv::resize(crop, crop, cv::Size(opt_.crop_size, opt_.crop_size), 0, 0, cv::INTER_AREA);
  cv::Mat rgb;
  cv::cvtColor(crop, rgb, cv::COLOR_BGR2RGB);
  rgb.convertTo(rgb, CV_32F);
  frame_idx_++;
  return {rgb, scene_switched};
}

}  // namespace av_tse
