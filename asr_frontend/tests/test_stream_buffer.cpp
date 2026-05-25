#include <gtest/gtest.h>

#include "asr_frontend/stream_buffer_2d.hpp"

using asr_frontend::StreamingRingBuffer2D;

TEST(StreamBuffer2D, WindowStride) {
  const int sr = 16000;
  const float dw = 16320.f / 16000.f;
  const int stride = 16000;
  StreamingRingBuffer2D buf(sr, dw, 1, stride, std::nullopt);
  EXPECT_EQ(buf.window(), 16320);
  EXPECT_EQ(buf.stride(), 16000);
}

TEST(StreamBuffer2D, PushUntilReady) {
  const int sr = 16000;
  const float dw = 1.0f;
  StreamingRingBuffer2D buf(sr, dw, 2, std::nullopt, std::nullopt);
  const int w = buf.window();
  Eigen::MatrixXf x(2, w - 1);
  x.setRandom();
  buf.push(x);
  EXPECT_FALSE(buf.ready());
  Eigen::MatrixXf one(2, 1);
  one.setConstant(0.25f);
  buf.push(one);
  EXPECT_TRUE(buf.ready());
  Eigen::MatrixXf win = buf.get_next_window();
  EXPECT_EQ(win.rows(), 2);
  EXPECT_EQ(win.cols(), w);
}
