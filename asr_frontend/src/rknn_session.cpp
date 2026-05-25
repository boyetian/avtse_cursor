#include "asr_frontend/rknn_session.hpp"

#include <cstring>
#include <fstream>
#include <stdexcept>

namespace asr_frontend {

std::vector<char> RknnSession::load_file(const std::string& path) {
  std::ifstream ifs(path, std::ios::binary | std::ios::ate);
  if (!ifs) {
    throw std::runtime_error("RknnSession: cannot open model: " + path);
  }
  const auto size = ifs.tellg();
  if (size <= 0) {
    throw std::runtime_error("RknnSession: empty model file: " + path);
  }
  std::vector<char> buf(static_cast<size_t>(size));
  ifs.seekg(0);
  if (!ifs.read(buf.data(), size)) {
    throw std::runtime_error("RknnSession: read failed: " + path);
  }
  return buf;
}

RknnSession::RknnSession(const std::string& model_path) {
  std::vector<char> model = load_file(model_path);
  int ret = rknn_init(&ctx_, model.data(), static_cast<uint32_t>(model.size()), 0, nullptr);
  if (ret < 0) {
    throw std::runtime_error("RknnSession: rknn_init failed, ret=" + std::to_string(ret));
  }
  ret = rknn_query(ctx_, RKNN_QUERY_IN_OUT_NUM, &io_num_, sizeof(io_num_));
  if (ret != RKNN_SUCC) {
    rknn_destroy(ctx_);
    ctx_ = 0;
    throw std::runtime_error("RknnSession: rknn_query IN_OUT_NUM failed");
  }
  query_tensor_attrs();
}

RknnSession::~RknnSession() {
  if (ctx_ != 0) {
    rknn_destroy(ctx_);
    ctx_ = 0;
  }
}

void RknnSession::query_tensor_attrs() {
  input_attrs_.resize(io_num_.n_input);
  for (uint32_t i = 0; i < io_num_.n_input; ++i) {
    rknn_tensor_attr attr{};
    attr.index = i;
    const int ret = rknn_query(ctx_, RKNN_QUERY_INPUT_ATTR, &attr, sizeof(attr));
    if (ret != RKNN_SUCC) {
      throw std::runtime_error("RknnSession: query input attr failed");
    }
    input_attrs_[i] = attr;
  }
  output_attrs_.resize(io_num_.n_output);
  for (uint32_t i = 0; i < io_num_.n_output; ++i) {
    rknn_tensor_attr attr{};
    attr.index = i;
    const int ret = rknn_query(ctx_, RKNN_QUERY_OUTPUT_ATTR, &attr, sizeof(attr));
    if (ret != RKNN_SUCC) {
      throw std::runtime_error("RknnSession: query output attr failed");
    }
    output_attrs_[i] = attr;
  }
}

void RknnSession::run(const std::vector<rknn_input>& inputs,
                      std::vector<rknn_output>& outputs) {
  if (inputs.size() != io_num_.n_input) {
    throw std::runtime_error("RknnSession: input count mismatch");
  }
  if (outputs.size() != io_num_.n_output) {
    throw std::runtime_error("RknnSession: output count mismatch");
  }
  int ret = rknn_inputs_set(ctx_, static_cast<uint32_t>(inputs.size()),
                            const_cast<rknn_input*>(inputs.data()));
  if (ret < 0) {
    throw std::runtime_error("RknnSession: rknn_inputs_set failed, ret=" + std::to_string(ret));
  }
  ret = rknn_run(ctx_, nullptr);
  if (ret < 0) {
    throw std::runtime_error("RknnSession: rknn_run failed, ret=" + std::to_string(ret));
  }
  ret = rknn_outputs_get(ctx_, static_cast<uint32_t>(outputs.size()), outputs.data(), nullptr);
  if (ret < 0) {
    throw std::runtime_error("RknnSession: rknn_outputs_get failed, ret=" + std::to_string(ret));
  }
}

}  // namespace asr_frontend
