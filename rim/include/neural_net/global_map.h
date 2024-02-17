#pragma once

#include <vector>

#include "mesher/mesher.h"
#include "sub_map.h"
#include "utils/utils.h"

struct Tensor3DHash {
  static constexpr long sl =
      10000; // adjust for different scale to avoid hash conflicts.
  static constexpr long sl2 = sl * sl;
  std::size_t operator()(const torch::Tensor &idx) const {
    return static_cast<unsigned int>(
        (idx[0] + idx[1] * sl + idx[2] * sl2).item<long>());
  }
};

struct Tensor3DEqual {
  bool operator()(const torch::Tensor &l, const torch::Tensor &r) const {
    return (l == r).all().item<bool>();
  }
};

struct GlobalMap : torch::nn::Module {

  std::unordered_map<torch::Tensor, SubMap *, Tensor3DHash, Tensor3DEqual>
      submaps_;
  torch::nn::Sequential decoder_;

  torch::Tensor submap_size_, submap_size_index_, offsets_, offsets_index;

  std::thread main_thread;
  std::mutex buf_mutex;
  std::condition_variable buf_cond;

  std::queue<std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, int>>
      save_buf;

  int active_map_num_;
  torch::Tensor latest_submap_index_;

  std::shared_ptr<Mesher> p_mesher_;

  GlobalMap();

  void
  save_decoder(const torch::OrderedDict<std::string, torch::Tensor> &_decoder);

  void save(const torch::Tensor &_w_index, const torch::Tensor &_feats,
            const torch::Tensor &_confs, int _layer);

  void fetch(const torch::Tensor &_xyzs, int _layer, torch::Tensor &_feats,
             torch::Tensor &_confs);

  void fetch_index(const torch::Tensor &_w_index, int _layer,
                   torch::Tensor &_feats, torch::Tensor &_confs);

  torch::Tensor encoding(const torch::Tensor &_xyz);

  torch::Tensor get_sdf(const torch::Tensor &_xyz);

  torch::Tensor sdf_to_sigmoid_sdf(const torch::Tensor &_sdf);

  // meshing using global meshing function and output mesh without stitches
  void meshing(const ros::Publisher &_mesh_pub,
               const ros::Publisher &_mesh_color_pub,
               const std_msgs::Header &_header, const float &_res,
               const std::string &_uuid = "global_mesh");

  void voxeling(ros::Publisher &voxel_map_pub, std_msgs::Header &_header);

  void main_loop();

  void save_(const torch::Tensor &_w_index, const torch::Tensor &_feats,
             const torch::Tensor &_confs, int _layer);

  void freeze_old_maps();

  void freeze_all_maps();

  torch::Tensor get_conf_mask(const torch::Tensor &_xyzs, int _layer = 0);

  torch::Tensor get_conf_mask_index(const torch::Tensor &_w_index,
                                    int _layer = 0);

  torch::Tensor get_conf_points(const torch::Tensor &_xyzs, int _layer = 0);

  torch::Tensor fetch_attribute(const torch::Tensor &_xyzs, int _layer,
                                int _type);

  torch::Tensor fetch_attribute_index(const torch::Tensor &_w_index, int _layer,
                                      int _type);
};