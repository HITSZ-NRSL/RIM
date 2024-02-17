#include "local_map.h"
#include "llog/llog.h"
#include "neural_net/global_map.h"
#include "tinycolormap.hpp"

#include "marching_cube/marching_cube.h"
#include "params/params.h"
#include "utils/utils.h"
using namespace std;

LocalMap::LocalMap()
    : SubMap(torch::zeros(3, *p_device), k_x_min, k_x_max, k_y_min, k_y_max,
             k_z_min, k_z_max) {
  p_mesher_ = std::make_shared<Mesher>();

  register_torch_parameter();

  decoder_->push_back(torch::nn::Linear(k_input_dim, k_hidden_dim));
  decoder_->push_back(torch::nn::ReLU());
  decoder_->push_back(torch::nn::Linear(k_hidden_dim, k_hidden_dim));
  decoder_->push_back(torch::nn::ReLU());
  decoder_->push_back(torch::nn::Linear(k_hidden_dim, 1));

  decoder_ = register_module("decoder", decoder_);
  decoder_->to(*p_device);
}

torch::Tensor LocalMap::get_shift_block_index(const int &delta_voxel_x,
                                              const int &delta_voxel_y,
                                              const int &delta_voxel_z,
                                              const int &layer) {
  std::vector<torch::Tensor> vec_indexes;
  if (delta_voxel_x != 0) {
    torch::Tensor tmp_mesh;
    if (delta_voxel_x > 0) {
      tmp_mesh = utils::meshgrid_3d(0, delta_voxel_x, 0, y_voxel_nums_[layer],
                                    0, z_voxel_nums_[layer], 1, *p_device)
                     .view({-1, 3});
    } else {
      tmp_mesh =
          utils::meshgrid_3d(x_voxel_nums_[layer] + delta_voxel_x,
                             x_voxel_nums_[layer], 0, y_voxel_nums_[layer], 0,
                             z_voxel_nums_[layer], 1, *p_device)
              .view({-1, 3});
    }
    vec_indexes.emplace_back(tmp_mesh);
  }
  if (delta_voxel_y != 0) {
    torch::Tensor tmp_mesh;
    if (delta_voxel_y > 0) {
      tmp_mesh = utils::meshgrid_3d(0, x_voxel_nums_[layer], 0, delta_voxel_y,
                                    0, z_voxel_nums_[layer], 1, *p_device)
                     .view({-1, 3});
    } else {
      tmp_mesh = utils::meshgrid_3d(0, x_voxel_nums_[layer],
                                    y_voxel_nums_[layer] + delta_voxel_y,
                                    y_voxel_nums_[layer], 0,
                                    z_voxel_nums_[layer], 1, *p_device)
                     .view({-1, 3});
    }
    vec_indexes.emplace_back(tmp_mesh);
  }
  if (delta_voxel_z != 0) {
    torch::Tensor tmp_mesh;
    if (delta_voxel_z > 0) {
      tmp_mesh =
          utils::meshgrid_3d(0, x_voxel_nums_[layer], 0, y_voxel_nums_[layer],
                             0, delta_voxel_z, 1, *p_device)
              .view({-1, 3});
    } else {
      tmp_mesh =
          utils::meshgrid_3d(0, x_voxel_nums_[layer], 0, y_voxel_nums_[layer],
                             z_voxel_nums_[layer] + delta_voxel_z,
                             z_voxel_nums_[layer], 1, *p_device)
              .view({-1, 3});
    }
    vec_indexes.emplace_back(tmp_mesh);
  }
  return torch::cat({vec_indexes}, 0).round().to(torch::kLong);
}

void LocalMap::save_to_globalmap(GlobalMap &_global_map) {
  for (int l = 0; l < k_layer_num; ++l) {
    torch::Tensor decay_indexes =
        utils::meshgrid_3d(0, x_voxel_nums_[l], 0, y_voxel_nums_[l], 0,
                           z_voxel_nums_[l], 1, *p_device)
            .view({-1, 3})
            .round()
            .to(torch::kLong);

    torch::Tensor tmp_idx = get_idx(decay_indexes, l);

    // [n,1]
    torch::Tensor decay_conf = conf_maps_[l].index({tmp_idx});
    torch::Tensor mask = decay_conf > 0;
    decay_conf = decay_conf.index({mask});
    if (decay_conf.size(0) > 0) {
      // [n,feat_dim]
      tmp_idx = tmp_idx.index({mask});
      torch::Tensor decay_feat =
          feat_maps_[l].data().index({tmp_idx}).detach().clone();

      decay_indexes = decay_indexes.index({mask});
      torch::Tensor decay_w_index = index_to_world_index(decay_indexes, l);
      _global_map.save_(decay_w_index, decay_feat, decay_conf.clone(), l);
    }
  }
}

void LocalMap::slide_map(const torch::Tensor &delta_voxel,
                         GlobalMap &_global_map, int layer) {
  long shift = (delta_voxel * t_idx_coef_[layer]).sum(1).item<long>();

  int delta_voxel_x = delta_voxel[0][0].item<int>();
  int delta_voxel_y = delta_voxel[0][1].item<int>();
  int delta_voxel_z = delta_voxel[0][2].item<int>();
  /// save
  // [n,3]
  torch::Tensor decay_indexes =
      get_shift_block_index(delta_voxel_x, delta_voxel_y, delta_voxel_z, layer);

  torch::Tensor tmp_idx = get_idx(decay_indexes, layer);
  // [n,1]
  torch::Tensor decay_conf = conf_maps_[layer].index({tmp_idx});
  torch::Tensor mask = decay_conf > 0;
  decay_conf = decay_conf.index({mask});
  if (decay_conf.size(0) > 0) {
    // [n,feat_dim]
    tmp_idx = tmp_idx.index({mask});
    torch::Tensor decay_feat =
        feat_maps_[layer].data().index({tmp_idx}).detach().clone();

    decay_indexes = decay_indexes.index({mask});
    torch::Tensor decay_w_index = index_to_world_index(decay_indexes, layer);
    _global_map.save(decay_w_index, decay_feat, decay_conf.clone(), layer);
  }

  /// shift
  t_idx_shift_[layer][0] += shift;
  index_W_M_[layer] = index_W_M_[layer] + delta_voxel;

  /// pad
  torch::Tensor grow_indexes = get_shift_block_index(
      -delta_voxel_x, -delta_voxel_y, -delta_voxel_z, layer);

  torch::Tensor grow_w_index = index_to_world_index(grow_indexes, layer);
  torch::Tensor grow_feats, grow_confs;
  _global_map.fetch_index(grow_w_index, layer, grow_feats, grow_confs);
  pad_maps(grow_indexes, layer, grow_feats, grow_confs);
}

void LocalMap::move_to(const torch::Tensor &pos_W_B, GlobalMap &_global_map) {
  torch::Tensor delta_voxel =
      ((pos_W_B - t_pos_W_M_) / voxel_sizes[0]).round().to(torch::kLong);
  if (delta_voxel.any().item<bool>()) {
    // #pragma omp parallel for
    for (int layer = 0; layer < k_layer_num; ++layer) {
      torch::Tensor tmp_delta_voxel =
          (delta_voxel * pow(2, layer)).to(torch::kLong);
      if (tmp_delta_voxel.any().item<bool>()) {
        slide_map(tmp_delta_voxel, _global_map, layer);
      }
    }

    /// [l,1,3]
    t_pos_W_M_ = delta_voxel * voxel_sizes[0] + t_pos_W_M_;
    // t_pos_W_M_ = torch::cat({pos_W_M_}, 0).view({k_layer_num, 1, 3});
  }
}

void LocalMap::unfreeze_net() {
  auto param_pairs = this->named_parameters();
  for (auto &param_pair : param_pairs) {
    param_pair.value().set_requires_grad(true);
  }
}

void LocalMap::freeze_net() {
  auto param_pairs = this->named_parameters();
  for (auto &param_pair : param_pairs) {
    param_pair.value().set_requires_grad(false);
  }
}

void LocalMap::freeze_decoder() {
  auto param_pairs = this->named_parameters();
  for (auto &param_pair : param_pairs) {
    // compare the first 7 char of key with "decoder"
    if (param_pair.key().substr(0, 7) == "decoder") {
      param_pair.value().set_requires_grad(false);
    }
  }
}

torch::Tensor LocalMap::encoding(const torch::Tensor &xyz, bool _update_conf) {
  /// [l,n,3]
  torch::Tensor xyz_loc = (xyz - t_pos_W_M_).unsqueeze(0) / t_voxel_sizes;
  /// [l,n,3]
  torch::Tensor xyz_voxel = xyz_loc.floor().to(torch::kLong);

  /// [l,n,3]
  torch::Tensor xyz_weight = (xyz_loc - xyz_voxel).view({-1, 3});
  // [l*n,8,1]
  torch::Tensor xyz_inter_coef =
      utils::cal_tri_inter_coef(xyz_weight).view({-1, 8, 1});

  /// [l*n,3]
  xyz_voxel = (xyz_voxel + t_pos_A_M_).view({-1, 3});

  // [l,n*8,3]
  torch::Tensor xyz_vertex =
      utils::get_verteices(xyz_voxel).view({k_layer_num, -1, 3});

  // [l,n*8]
  torch::Tensor xyz_vertex_idx = get_idx(xyz_vertex);

  if (_update_conf) {
    // #pragma omp parallel for
    for (int l = 0; l < k_layer_num; ++l) {
      conf_maps_[l].index_put_({xyz_vertex_idx.select(0, l)}, 1);
    }
  }

  //[l*n,8,feat_dim]
  torch::Tensor xyz_vertex_feat = feat_maps_[0]
                                      .index({xyz_vertex_idx.select(0, 0)})
                                      .view({-1, 8, k_feat_dim});
  for (int l = 1; l < k_layer_num; ++l) {
    xyz_vertex_feat =
        torch::cat({xyz_vertex_feat, feat_maps_[l]
                                         .index({xyz_vertex_idx.select(0, l)})
                                         .view({-1, 8, k_feat_dim})},
                   0);
  }

  //[l,n,feat_dim]
  torch::Tensor xyz_feat = (xyz_inter_coef * xyz_vertex_feat)
                               .sum(1)
                               .view({k_layer_num, -1, k_feat_dim});
  if (k_multifeat_type == 0) {
    torch::Tensor tmp_xyz_feat = xyz_feat.select(0, 0);
    for (int l = 1; l < k_layer_num; ++l) {
      tmp_xyz_feat = torch::cat({tmp_xyz_feat, xyz_feat.select(0, l)}, 1);
    }
    xyz_feat = tmp_xyz_feat;
  } else {
    xyz_feat = xyz_feat.sum(0);
  }

  if (k_positional_encoding) {
    xyz_feat = torch::cat({xyz_feat, utils::positional_encode(xyz)}, -1);
  }

  return xyz_feat;
}

/**
 * @description:
 * @param {Tensor} &xyz
 * @param {bool} _skip_unconf
 * @return {Tensor} sdf
 */
torch::Tensor LocalMap::get_sdf(const torch::Tensor &xyz, bool _update_conf) {
  static auto p_t_encoding = llog::CreateTimer("  encoding");
  p_t_encoding->tic();

  torch::Tensor xyz_feat = encoding(xyz, _update_conf);

  p_t_encoding->toc_avg();
  static auto p_t_forward = llog::CreateTimer("  forward");
  p_t_forward->tic();

  torch::Tensor xyz_sdf = decoder_->forward(xyz_feat);

  p_t_forward->toc_avg();
  return xyz_sdf;
}

torch::Tensor LocalMap::sdf_to_sigmoid_sdf(const torch::Tensor &sdf) {
  return 1 / (1 + torch::exp(sdf / k_bce_sigma));
}

void LocalMap::meshing_(ros::Publisher &mesh_pub,
                        ros::Publisher &mesh_color_pub,
                        std_msgs::Header &header, float _res, bool _save,
                        const torch::Tensor &xyz, const std::string &uuid) {
  if (k_eval_mode || _save) {
    meshing_map(mesh_pub, mesh_color_pub, header, _res, _save, uuid);
  } else {
    meshing_xyz(mesh_pub, mesh_color_pub, header, _res, uuid, xyz);
  }
}

torch::Tensor LocalMap::voxelized_xyz(const torch::Tensor &_pos,
                                      const torch::Tensor &_xyz, int _eval_mode,
                                      float _res, float _slice_height) const {
  torch::Tensor xyz_voxelized;
  if (_eval_mode) {
    static torch::Tensor vis_mesh_xyz_slice = utils::meshgrid_3d_slice(
        xyz_min_[0].item<float>(), xyz_max_[0].item<float>(),
        xyz_min_[1].item<float>(), xyz_max_[1].item<float>(),
        xyz_min_[2].item<float>(), xyz_max_[2].item<float>(), _res, *p_device,
        k_slice_height);
    xyz_voxelized = vis_mesh_xyz_slice + _pos.view({1, 1, 1, 3});
  } else {
    torch::Tensor bounded_mask = get_inrange_mask(_xyz, _res);
    torch::Tensor bounded_xyz = _xyz.index({bounded_mask});

    torch::Tensor voxel_index = (bounded_xyz / _res).floor().to(torch::kLong);
    voxel_index = std::get<0>(torch::unique_dim(voxel_index, 0));
    xyz_voxelized = voxel_index * _res;
  }
  return xyz_voxelized;
}

void LocalMap::meshing_xyz(ros::Publisher &_mesh_pub,
                           ros::Publisher &_mesh_color_pub,
                           std_msgs::Header &_header, float _res,
                           const std::string &_uuid,
                           const torch::Tensor &_xyz) {
  torch::GradMode::set_enabled(false);
  torch::Tensor xyz_voxelized =
      voxelized_xyz(t_pos_W_M_, _xyz, false, _res).view({-1, 3});
  if (xyz_voxelized.size(0) == 0) {
    cout << "Marching cube: no point!"
         << "\n";
    return;
  }
  long batch_meshing_num = xyz_voxelized.size(0) / k_vis_batch_pt_num + 1;
  for (int i = 0; i < batch_meshing_num; ++i) {
    long start = i * k_vis_batch_pt_num;
    long end = (i + 1) * k_vis_batch_pt_num;
    if (i == batch_meshing_num - 1) {
      end = end > xyz_voxelized.size(0) ? xyz_voxelized.size(0) : end;
    }
    if (end == start)
      continue;
    torch::Tensor tmp_xyz_voxelized =
        xyz_voxelized.index({torch::indexing::Slice(start, end)});

    // TODO: redundant computation
    // [n,8,3]
    torch::Tensor xyz_vertex =
        utils::get_width_verteices(tmp_xyz_voxelized, _res);
    // http://paulbourke.net/geometry/polygonise/
    // WARN: delicate order!
    /// [n*8,3]
    xyz_vertex =
        torch::stack({xyz_vertex.select(1, 0), xyz_vertex.select(1, 2),
                      xyz_vertex.select(1, 6), xyz_vertex.select(1, 4),
                      xyz_vertex.select(1, 1), xyz_vertex.select(1, 3),
                      xyz_vertex.select(1, 7), xyz_vertex.select(1, 5)},
                     1)
            .view({-1, 3});

    torch::Tensor voxel_vertex_sdf;

    auto xyz_sdf = get_sdf(xyz_vertex, false);
    voxel_vertex_sdf = xyz_sdf.view({-1, 8});

    /// [n,3,3]
    torch::Tensor voxel_face_xyz = marching_cube(voxel_vertex_sdf, xyz_vertex);
    if (voxel_face_xyz.size(0) == 0)
      continue;

    pub_mesh(_mesh_pub, _mesh_color_pub, voxel_face_xyz, _header, _uuid);
  }
}

void LocalMap::meshing_map(ros::Publisher &_mesh_pub,
                           ros::Publisher &_mesh_color_pub,
                           std_msgs::Header &_header, float _res, bool _save,
                           const std::string &_uuid) {
  torch::GradMode::set_enabled(false);

  if (_save) {
    p_mesher_->vec_face_xyz_.clear();
  }

  float x_res = (xyz_max_[0].item<float>() - xyz_min_[0].item<float>()) / _res;
  float y_res = (xyz_max_[1].item<float>() - xyz_min_[1].item<float>()) / _res;
  float z_res = (xyz_max_[2].item<float>() - xyz_min_[2].item<float>()) / _res;
  float yz_res = y_res * z_res;

  int x_step = k_vis_batch_pt_num / yz_res + 1;
  int steps = x_res / x_step + 1;
  float step_size = x_step * _res;

  for (int i = 0; i < steps; ++i) {
    float start = i * step_size + xyz_min_[0].item<float>();
    float end = start + step_size;
    if (i == steps - 1) {
      end = end > xyz_max_[0].item<float>() ? xyz_max_[0].item<float>() : end;
    }
    if (end == start)
      break;

    torch::Tensor mesh_xyz =
        utils::meshgrid_3d(start, end + _res, xyz_min_[1].item<float>(),
                           xyz_max_[1].item<float>() + _res,
                           xyz_min_[2].item<float>(),
                           xyz_max_[2].item<float>() + _res, _res, *p_device) +
        t_pos_W_M_.view({1, 1, 1, 3});
    long x_num = mesh_xyz.size(0);
    long y_num = mesh_xyz.size(1);
    long z_num = mesh_xyz.size(2);
    mesh_xyz = mesh_xyz.view({-1, 3});

    torch::Tensor mesh_mask = get_conf_mask(mesh_xyz, k_layer_num - 1);
    torch::Tensor mask_xyz_sdf = get_sdf(mesh_xyz.index({mesh_mask}), false);
    torch::Tensor mesh_sdf = torch::zeros({mesh_xyz.size(0), 1}).to(*p_device);
    mesh_sdf.index_put_({mesh_mask}, mask_xyz_sdf);

    /// [n,3,3]
    auto face_xyz = xyz_sdf_mask_to_face(mesh_xyz, mesh_sdf, mesh_mask, x_num,
                                         y_num, z_num);
    if (face_xyz.size(0) == 0)
      continue;

    pub_mesh(_mesh_pub, _mesh_color_pub, face_xyz, _header, _uuid);
    if (_save) {
      p_mesher_->vec_face_xyz_.emplace_back(face_xyz.cpu());
    }
  }
}

visualization_msgs::Marker LocalMap::get_sdf_map_(const torch::Tensor &_pos,
                                                  const torch::Tensor &_xyz,
                                                  float _res,
                                                  float _slice_height,
                                                  float _truncated_distance) {
  torch::Tensor xyz_voxelized =
      voxelized_xyz(_pos, _xyz, true, _res, _slice_height).view({-1, 3});

  torch::Tensor mask = get_conf_mask(xyz_voxelized, 0);
  xyz_voxelized = xyz_voxelized.index({mask});

  if (xyz_voxelized.size(0) == 0)
    return {};

  auto xyz_sdf = get_sdf(xyz_voxelized, false).to(torch::kCPU);
  xyz_voxelized = xyz_voxelized.to(torch::kCPU);

  auto xyz_sdf_a = xyz_sdf.accessor<float, 1>();
  auto xyz_voxelized_a = xyz_voxelized.accessor<float, 2>();

  visualization_msgs::Marker marker;
  marker.type = visualization_msgs::Marker::CUBE_LIST;
  marker.action = visualization_msgs::Marker::ADD;
  marker.scale.x = _res;
  marker.scale.y = _res;
  marker.scale.z = _res;
  marker.points.resize(xyz_sdf.size(0));
  marker.colors.resize(xyz_sdf.size(0));
#pragma omp parallel for
  for (int i = 0; i < xyz_sdf.size(0); ++i) {
    geometry_msgs::Point p;
    p.x = xyz_voxelized_a[i][0];
    p.y = xyz_voxelized_a[i][1];
    p.z = xyz_voxelized_a[i][2];
    marker.points[i] = p;

    auto color = tinycolormap::GetColor(
        xyz_sdf_a[i] > _truncated_distance ? 1.0
                                           : xyz_sdf_a[i] / _truncated_distance,
        tinycolormap::ColormapType::Heat);

    marker.colors[i].a = 1.0;
    marker.colors[i].r = color.r();
    marker.colors[i].g = color.g();
    marker.colors[i].b = color.b();
  }
  return marker;
}
