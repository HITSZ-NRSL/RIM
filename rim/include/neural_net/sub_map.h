#pragma once
#include <torch/torch.h>

struct SubMap : torch::nn::Module {
  std::vector<torch::Tensor> feat_maps_;

  std::vector<torch::Tensor> conf_maps_; // 0: not visited, 1: visited
  std::vector<torch::Tensor> index_A_M_; // pos: Body(index) to Array
  std::vector<torch::Tensor> index_W_M_; // pos: Map(index) to World
  torch::Tensor t_pos_W_M_;              // pos: Map(xyz) to World
  torch::Tensor t_pos_A_M_;
  torch::Tensor t_idx_coef_;
  torch::Tensor t_idx_shift_;
  torch::Tensor t_voxel_nums_;
  torch::Tensor xyz_max_, xyz_min_;

  std::vector<long> x_voxel_nums_, y_voxel_nums_, z_voxel_nums_, voxel_nums_;

  bool active_;
  std::mutex state_mutex_;

  SubMap(const torch::Tensor &_pos_W_M, float _x_min, float _x_max,
         float _y_min, float _y_max, float _z_min, float _z_max);

  void register_torch_parameter();

  void activate();

  void freeze();

  void pad_maps(const torch::Tensor &_xyz_index, int _layer);

  void pad_maps(const torch::Tensor &_xyz_index, int &_layer,
                const torch::Tensor &_feats, const torch::Tensor &_confs);

  torch::Tensor get_inrange_mask(const torch::Tensor &_xyz,
                                 float padding = 0.0) const;

  void get_intersect_point(const torch::Tensor &_points,
                           const torch::Tensor &_rays, torch::Tensor &z_nears,
                           torch::Tensor &z_fars, torch::Tensor &mask_intersect,
                           float padding = 0.0) const;
  /**
   * @description:
   * @param {Tensor} &xyz_index
   * @param {int} layer: -1 for all layers
   * @return {*}
   */
  torch::Tensor get_idx(const torch::Tensor &xyz_index, int layer = -1) const;

  torch::Tensor xyz_to_index(const torch::Tensor &xyz, int layer) const;

  torch::Tensor index_to_xyz(const torch::Tensor &_index, int layer) const;

  torch::Tensor index_to_world_index(const torch::Tensor &_index,
                                     int layer) const;

  torch::Tensor world_index_to_index(const torch::Tensor &_w_index,
                                     int layer) const;

  torch::Tensor world_index_to_idx(const torch::Tensor &_w_index,
                                   int layer) const;

  torch::Tensor xyz_to_idx(const torch::Tensor &xyz, int layer) const;

  torch::Tensor get_conf_mask(const torch::Tensor &xyz, int layer = 0);

  torch::Tensor get_conf_points(const torch::Tensor &xyz, int layer = 0);
};