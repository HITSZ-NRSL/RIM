#pragma once

#include "neural_net/global_map.h"

struct LocalMap : SubMap {
  torch::nn::Sequential decoder_;

  std::shared_ptr<Mesher> p_mesher_;

  LocalMap();

  torch::Tensor get_shift_block_index(const int &delta_voxel_x,
                                      const int &delta_voxel_y,
                                      const int &delta_voxel_z,
                                      const int &layer);

  void save_to_globalmap(GlobalMap &_global_map);

  void slide_map(const torch::Tensor &delta_voxel, GlobalMap &_global_map,
                 int layer);

  void move_to(const torch::Tensor &pos_W_B, GlobalMap &_global_map);

  void unfreeze_net();

  void freeze_net();

  void freeze_decoder();

  torch::Tensor encoding(const torch::Tensor &xyz, bool _update_conf = true);

  torch::Tensor get_sdf(const torch::Tensor &xyz, bool _update_conf = true);

  torch::Tensor sdf_to_sigmoid_sdf(const torch::Tensor &sdf);

  void meshing_(ros::Publisher &mesh_pub, ros::Publisher &mesh_color_pub,
                std_msgs::Header &header, float _res, bool _save = false,
                const torch::Tensor &xyz = torch::Tensor(),
                const std::string &uuid = "mesh_map");

  torch::Tensor voxelized_xyz(const torch::Tensor &_pos,
                              const torch::Tensor &_xyz, int _eval_mode,
                              float _res, float _slice_height = -1) const;

  void meshing_xyz(ros::Publisher &_mesh_pub, ros::Publisher &_mesh_color_pub,
                   std_msgs::Header &_header, float _res,
                   const std::string &_uuid = "mesh",
                   const torch::Tensor &_xyz = torch::Tensor());

  void meshing_map(ros::Publisher &_mesh_pub, ros::Publisher &_mesh_color_pub,
                   std_msgs::Header &_header, float _res, bool _save = false,
                   const std::string &_uuid = "mesh");

  visualization_msgs::Marker get_sdf_map_(const torch::Tensor &_pos,
                                          const torch::Tensor &_xyz, float _res,
                                          float _slice_height = -1,
                                          float _truncated_distance = 1.0);
};