#include "sub_map.h"
#include "params/params.h"

using namespace std;

SubMap::SubMap(const torch::Tensor &_pos_W_M, float _x_min, float _x_max,
               float _y_min, float _y_max, float _z_min, float _z_max)
    : torch::nn::Module(), t_pos_W_M_(_pos_W_M.view({1, 3})) {

  /// [l,1,1]
  t_idx_shift_ = torch::zeros({k_layer_num, 1}, *p_device).to(torch::kLong);
  t_idx_coef_ = torch::ones({k_layer_num, 1, 3}, *p_device).to(torch::kLong);
  t_voxel_nums_ = torch::zeros({k_layer_num, 1}, *p_device).to(torch::kLong);

  for (int i = 0; i < k_layer_num; ++i) {
    x_voxel_nums_.emplace_back(ceil((_x_max - _x_min) / voxel_sizes[i]));
    y_voxel_nums_.emplace_back(ceil((_y_max - _y_min) / voxel_sizes[i]));
    z_voxel_nums_.emplace_back(ceil((_z_max - _z_min) / voxel_sizes[i]));

    t_idx_coef_[i][0][1] = x_voxel_nums_[i];
    t_idx_coef_[i][0][2] = x_voxel_nums_[i] * y_voxel_nums_[i];

    voxel_nums_.emplace_back(x_voxel_nums_[i] * y_voxel_nums_[i] *
                             z_voxel_nums_[i]);
    t_voxel_nums_[i][0] = voxel_nums_[i];

    index_A_M_.emplace_back(
        (torch::tensor({-_x_min / voxel_sizes[i], -_y_min / voxel_sizes[i],
                        -_z_min / voxel_sizes[i]},
                       *p_device))
            .round()
            .to(torch::kLong));

    index_W_M_.emplace_back(
        (t_pos_W_M_ / voxel_sizes[i]).round().to(torch::kLong));

    if (k_zero_init) {
      feat_maps_.emplace_back(
          torch::zeros({voxel_nums_[i], k_feat_dim}, *p_device));
    } else {
      feat_maps_.emplace_back(
          torch::randn({voxel_nums_[i], k_feat_dim}, *p_device));
    }

    conf_maps_.emplace_back(
        torch::zeros({voxel_nums_[i]}, *p_device).to(torch::kInt));
  }

  /// [l,1,3]
  t_pos_A_M_ = torch::cat({index_A_M_}, 0).view({k_layer_num, 1, 3});

  xyz_min_ = -index_A_M_[0] * voxel_sizes[0];
  xyz_max_ = xyz_min_ + voxel_sizes[0] * torch::tensor({x_voxel_nums_[0] - 1,
                                                        y_voxel_nums_[0] - 1,
                                                        z_voxel_nums_[0] - 1},
                                                       *p_device);

  /// visualization init
  xyz_min_ = xyz_min_ + voxel_sizes[0];
  xyz_max_ = xyz_max_ - voxel_sizes[0];
  active_ = true;
}

void SubMap::register_torch_parameter() {
  for (int i = 0; i < k_layer_num; ++i) {
    feat_maps_[i] = register_parameter("feat_densemap_" + std::to_string(i),
                                       feat_maps_[i], true);
  }
}

void SubMap::activate() {
  state_mutex_.lock();
  this->to(*p_device);
  active_ = true;
  state_mutex_.unlock();
}

void SubMap::freeze() {
  state_mutex_.lock();
  this->to(torch::kCPU);
  active_ = false;
  state_mutex_.unlock();
}

void SubMap::pad_maps(const torch::Tensor &_xyz_index, int _layer) {
  torch::Tensor tmp_idx = get_idx(_xyz_index, _layer);
  feat_maps_[_layer].data().index_put_({tmp_idx}, 0.0);
  conf_maps_[_layer].index_put_({tmp_idx}, 0.0);
}

void SubMap::pad_maps(const torch::Tensor &_xyz_index, int &_layer,
                      const torch::Tensor &_feats,
                      const torch::Tensor &_confs) {
  torch::Tensor tmp_idx = get_idx(_xyz_index, _layer);
  feat_maps_[_layer].data().index_put_({tmp_idx}, _feats.data());
  conf_maps_[_layer].index_put_({tmp_idx}, _confs);
}

torch::Tensor SubMap::get_inrange_mask(const torch::Tensor &_xyz,
                                       float padding) const {
  torch::Tensor xyz_tmp = _xyz - t_pos_W_M_;
  torch::Tensor max_mask = xyz_tmp < (xyz_max_ - padding - 1e-6);
  torch::Tensor min_mask = xyz_tmp > (xyz_min_ + padding + 1e-6);
  torch::Tensor mask = max_mask & min_mask;
  return mask.all(1);
}

void SubMap::get_intersect_point(const torch::Tensor &_points,
                                 const torch::Tensor &_rays,
                                 torch::Tensor &z_nears, torch::Tensor &z_fars,
                                 torch::Tensor &mask_intersect,
                                 float padding) const {
  // Force the parallel rays to intersect with plane, and they will be removed
  // by sanity check, add small number to avoid judgement
  torch::Tensor mask_parallels = _rays == 0;
  torch::Tensor tmp_rays =
      _rays.index_put({mask_parallels}, _rays.index({mask_parallels}) + 1e-6);

  torch::Tensor tmp_z_nears =
      (t_pos_W_M_ + xyz_min_ - _points + padding) / tmp_rays;
  torch::Tensor tmp_z_fars =
      (t_pos_W_M_ + xyz_max_ - _points - padding) / tmp_rays;

  // Make sure near is closer than far
  torch::Tensor mask_exchange = tmp_z_nears > tmp_z_fars;
  z_nears =
      tmp_z_nears.index_put({mask_exchange}, tmp_z_fars.index({mask_exchange}));
  z_fars =
      tmp_z_fars.index_put({mask_exchange}, tmp_z_nears.index({mask_exchange}));

  z_nears = get<0>(torch::max(z_nears, 1));
  z_fars = get<0>(torch::min(z_fars, 1));

  // check if intersect
  mask_intersect = z_nears < z_fars;
}

/**
 * @description:
 * @param {Tensor} &xyz_index
 * @param {int} layer: -1 for all layers
 * @return {*}
 */
torch::Tensor SubMap::get_idx(const torch::Tensor &xyz_index, int layer) const {
  if (layer == -1) {
    /* xyz_index: [l,n,3] */
    return ((xyz_index * t_idx_coef_).sum(-1) + t_idx_shift_) % t_voxel_nums_;
  } else {
    /* xyz_index: [n,3] */
    return ((xyz_index * t_idx_coef_[layer]).sum(-1) + t_idx_shift_[layer][0]) %
           t_voxel_nums_[layer][0];
  }
}

torch::Tensor SubMap::xyz_to_index(const torch::Tensor &xyz, int layer) const {
  return ((xyz - t_pos_W_M_) / voxel_sizes[layer]).floor().to(torch::kLong) +
         index_A_M_[layer];
}

torch::Tensor SubMap::index_to_xyz(const torch::Tensor &_index,
                                   int layer) const {
  return ((_index - index_A_M_[layer]) * voxel_sizes[layer] + t_pos_W_M_);
}

torch::Tensor SubMap::index_to_world_index(const torch::Tensor &_index,
                                           int layer) const {
  return _index - index_A_M_[layer] + index_W_M_[layer];
}

torch::Tensor SubMap::world_index_to_index(const torch::Tensor &_w_index,
                                           int layer) const {
  return _w_index - index_W_M_[layer] + index_A_M_[layer];
}

torch::Tensor SubMap::world_index_to_idx(const torch::Tensor &_w_index,
                                         int layer) const {
  return get_idx(world_index_to_index(_w_index, layer), layer);
}

torch::Tensor SubMap::xyz_to_idx(const torch::Tensor &xyz, int layer) const {
  return get_idx(xyz_to_index(xyz, layer), layer);
}

torch::Tensor SubMap::get_conf_mask(const torch::Tensor &xyz, int layer) {
  return conf_maps_[layer].index({xyz_to_idx(xyz, layer)}) > 0;
}

torch::Tensor SubMap::get_conf_points(const torch::Tensor &xyz, int layer) {
  return xyz.index({get_conf_mask(xyz, layer)});
}
