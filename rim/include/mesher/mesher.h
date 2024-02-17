#pragma once

#include "marching_cube/marching_cube.h"
#include <torch/torch.h>
#include <vector>

void face_to_ply(const torch::Tensor &face_xyz, const std::string &filename);

// intake xyz, get_conf_mask(), get_sdf() to get the mesh
torch::Tensor
xyz_sdf_mask_to_face(torch::Tensor &_xyz, torch::Tensor &_sdf,
                     torch::Tensor &_mask, long x_num, long y_num, long z_num,
                     float _isovalue = 0.0);

class Mesher {
public:
  Mesher();

  std::vector<torch::Tensor> vec_face_xyz_;

  void save_mesh();
};