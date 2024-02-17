#pragma once

#include <torch/torch.h>

#include "utils/utils.h"

using namespace std;

struct Odometry : torch::nn::Module {
  Odometry(torch::Device _device) : device(_device) {
    quat_ =
        register_parameter("quat", torch::tensor({1.0, 0.0, 0.0, 0.0}, device));
    pos_ = register_parameter("pos", torch::tensor({0.0, 0.0, 0.0}, device));
  }

  torch::Device device;
  torch::Tensor quat_, pos_;

  torch::Tensor quat() { return quat_; }
  torch::Tensor rot() { return quat_to_rot(quat_); }
  torch::Tensor pos() { return pos_; }
  torch::Tensor rot_pos() {
    auto rotation = quat_to_rot(quat_);
    auto rot_pos = torch::cat({rotation, pos_.unsqueeze(1)}, 1);
    return rot_pos;
  }
};