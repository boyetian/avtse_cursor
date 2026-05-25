#pragma once

#include <rknn_api.h>

#include <cstdint>
#include <string>
#include <vector>

namespace asr_frontend {

/// Minimal RKNN runtime wrapper (init / query / run / release).
class RknnSession {
 public:
  explicit RknnSession(const std::string& model_path);
  ~RknnSession();

  RknnSession(const RknnSession&) = delete;
  RknnSession& operator=(const RknnSession&) = delete;

  rknn_context ctx() const { return ctx_; }
  const rknn_input_output_num& io_num() const { return io_num_; }
  const std::vector<rknn_tensor_attr>& input_attrs() const { return input_attrs_; }
  const std::vector<rknn_tensor_attr>& output_attrs() const { return output_attrs_; }

  /// Runs inference; `outputs` must be sized to `io_num_.n_output` with `want_float` set as needed.
  void run(const std::vector<rknn_input>& inputs, std::vector<rknn_output>& outputs);

 private:
  static std::vector<char> load_file(const std::string& path);
  void query_tensor_attrs();

  rknn_context ctx_{0};
  rknn_input_output_num io_num_{};
  std::vector<rknn_tensor_attr> input_attrs_;
  std::vector<rknn_tensor_attr> output_attrs_;
};

}  // namespace asr_frontend
