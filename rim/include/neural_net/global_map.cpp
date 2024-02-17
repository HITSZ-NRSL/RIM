#include "global_map.h"
#include <iostream>

#include <omp.h>

#include "params/params.h"
#include <tinyply/tinyply.h>

using namespace std;

GlobalMap::GlobalMap() {
  p_mesher_ = std::make_shared<Mesher>();

  submap_size_ = torch::tensor(
      {{k_x_max - k_x_min, k_y_max - k_y_min, k_z_max - k_z_min}}, *p_device);
  submap_size_index_ = (submap_size_.view({1, 1, 3}) / t_voxel_sizes).ceil();

  offsets_ = torch::tensor({{-k_x_min, -k_y_min, -k_z_min}}, *p_device);
  offsets_index =
      (offsets_.view({1, 1, 3}) / t_voxel_sizes).round().to(torch::kLong);

  decoder_->push_back(torch::nn::Linear(k_input_dim, k_hidden_dim));
  decoder_->push_back(torch::nn::ReLU());
  decoder_->push_back(torch::nn::Linear(k_hidden_dim, k_hidden_dim));
  decoder_->push_back(torch::nn::ReLU());
  decoder_->push_back(torch::nn::Linear(k_hidden_dim, 1));

  decoder_ = register_module("decoder", decoder_);
  decoder_->to(*p_device);

  active_map_num_ = 0;

  main_thread = std::thread(&GlobalMap::main_loop, this);
}

void GlobalMap::save_decoder(
    const torch::OrderedDict<std::string, torch::Tensor> &_decoder) {
  torch::GradMode::set_enabled(false);
  auto params = this->named_parameters();
  for (auto &pair : _decoder) {
    if (pair.key().substr(0, 7) == "decoder") {
      params[pair.key()].copy_(pair.value());
    }
  }
}

void GlobalMap::save(const torch::Tensor &_w_index, const torch::Tensor &_feats,
                     const torch::Tensor &_confs, int _layer) {
  buf_mutex.lock();
  save_buf.emplace(_w_index, _feats, _confs, _layer);
  buf_mutex.unlock();
  buf_cond.notify_one();
}

void GlobalMap::fetch(const torch::Tensor &_xyzs, int _layer,
                      torch::Tensor &_feats, torch::Tensor &_confs) {
  auto submap_indices =
      ((_xyzs + offsets_) / submap_size_).floor().to(torch::kLong);
  auto submap_indices_uniq = (get<0>(torch::unique_dim(submap_indices, 0)));

  if (k_zero_init) {
    _feats = torch::zeros({_xyzs.size(0), k_feat_dim}, *p_device);
  } else {
    _feats = torch::randn({_xyzs.size(0), k_feat_dim}, *p_device);
  }
  _confs = torch::zeros({_xyzs.size(0)}, *p_device).to(torch::kInt);
  // #pragma omp parallel for reduction(+ : active_map_num_)
  for (int i = 0; i < submap_indices_uniq.size(0); i++) {
    auto submap_index = submap_indices_uniq[i];
    auto it = submaps_.find(submap_index);
    if (it == submaps_.end()) {
      continue;
    } else if (!it->second->active_) {
      it->second->activate();
      ++active_map_num_;
    }

    auto mask = submap_indices == submap_index.unsqueeze(0);
    mask = mask.all(1);

    auto voxel_xyzs = _xyzs.index({mask});
    auto submap_idx = it->second->xyz_to_idx(voxel_xyzs, _layer);
    if ((submap_idx.min() < 0).item<bool>()) {
      cout << "fetch min error!!!"
           << "\n";
      cout << submap_idx.min() << "\n";
    }
    if ((submap_idx.max() >= it->second->feat_maps_[_layer].size(0))
            .item<bool>()) {
      cout << "fetch max error!!!"
           << "\n";
      cout << submap_idx.max() << "\n";
    }
    _feats.data().index_put_(
        {mask}, it->second->feat_maps_[_layer].data().index({submap_idx}));
    _confs.index_put_({mask},
                      it->second->conf_maps_[_layer].index({submap_idx}));
  }
}

void GlobalMap::fetch_index(const torch::Tensor &_w_index, int _layer,
                            torch::Tensor &_feats, torch::Tensor &_confs) {
  auto submap_indices =
      ((_w_index + offsets_index[_layer]) / submap_size_index_[_layer])
          .floor()
          .to(torch::kLong);
  auto submap_indices_uniq = (get<0>(torch::unique_dim(submap_indices, 0)));

  if (k_zero_init) {
    _feats = torch::zeros({_w_index.size(0), k_feat_dim}, *p_device);
  } else {
    _feats = torch::randn({_w_index.size(0), k_feat_dim}, *p_device);
  }
  _confs = torch::zeros({_w_index.size(0)}, *p_device).to(torch::kInt);
  // #pragma omp parallel for
  for (int i = 0; i < submap_indices_uniq.size(0); i++) {
    auto submap_index = submap_indices_uniq[i];
    auto it = submaps_.find(submap_index);
    if (it == submaps_.end()) {
      continue;
    } else if (!it->second->active_) {
      it->second->activate();
      ++active_map_num_;
    }

    auto mask = submap_indices == submap_index.unsqueeze(0);
    mask = mask.all(1);

    auto voxel_index = _w_index.index({mask});
    auto submap_idx = it->second->world_index_to_idx(voxel_index, _layer);

    _feats.data().index_put_(
        {mask}, it->second->feat_maps_[_layer].data().index({submap_idx}));
    _confs.index_put_({mask},
                      it->second->conf_maps_[_layer].index({submap_idx}));
  }
}

torch::Tensor GlobalMap::encoding(const torch::Tensor &_xyz) {
  /// [l,n*8,3]
  torch::Tensor xyz_loc = _xyz.unsqueeze(0) / t_voxel_sizes;
  /// [l,n*8,3]
  auto xyz_voxel = xyz_loc.floor().to(torch::kLong);
  /// [l*n*8,3]
  auto xyz_weight = (xyz_loc - xyz_voxel).view({-1, 3});
  // [l*n*8,8,1]
  auto xyz_inter_coef = utils::cal_tri_inter_coef(xyz_weight).view({-1, 8, 1});
  /// [l*n*8,3]
  xyz_voxel = xyz_voxel.view({-1, 3});
  // [l,n*8*8,3]
  auto xyz_voxel_vertex =
      utils::get_verteices(xyz_voxel).view({k_layer_num, -1, 3});

  // [l,n*8*8,3]
  auto xyz_vertex_feat =
      fetch_attribute_index(xyz_voxel_vertex.select(0, 0), 0, 0)
          .view({-1, 8, k_feat_dim});
  for (int l = 1; l < k_layer_num; ++l) {
    xyz_vertex_feat =
        torch::cat({xyz_vertex_feat,
                    fetch_attribute_index(xyz_voxel_vertex.select(0, l), l, 0)
                        .view({-1, 8, k_feat_dim})},
                   0);
  }
  //[l,n*8,feat_dim]
  auto xyz_feat = (xyz_inter_coef * xyz_vertex_feat)
                      .sum(1)
                      .view({k_layer_num, -1, k_feat_dim});
  if (k_multifeat_type == 0) {
    auto tmp_xyz_feat = xyz_feat.select(0, 0);
    for (int l = 1; l < k_layer_num; ++l) {
      tmp_xyz_feat = torch::cat({tmp_xyz_feat, xyz_feat.select(0, l)}, 1);
    }
    xyz_feat = tmp_xyz_feat;
  } else {
    xyz_feat = xyz_feat.sum(0);
  }

  if (k_positional_encoding) {
    xyz_feat = torch::cat({xyz_feat, utils::positional_encode(_xyz)}, -1);
  }
  return xyz_feat;
}

torch::Tensor GlobalMap::sdf_to_sigmoid_sdf(const torch::Tensor &_sdf) {
  return 1 / (1 + torch::exp(_sdf / k_bce_sigma));
}

torch::Tensor GlobalMap::get_sdf(const torch::Tensor &_xyz) {
  auto xyz_feat = encoding(_xyz);
  return decoder_->forward(xyz_feat);
}

// meshing using global meshing function and output mesh without stitches
void GlobalMap::meshing(const ros::Publisher &_mesh_pub,
                        const ros::Publisher &_mesh_color_pub,
                        const std_msgs::Header &_header, const float &_res,
                        const string &_uuid) {
  cout << "Start meshing global map..."
       << "\n";
  torch::GradMode::set_enabled(false);
  p_mesher_->vec_face_xyz_.clear();

  auto x_res = (k_x_max - k_x_min) / _res;
  auto y_res = (k_y_max - k_y_min) / _res;
  auto z_res = (k_z_max - k_z_min) / _res;
  auto yz_res = y_res * z_res;

  int x_step = k_vis_batch_pt_num / yz_res + 1;
  int steps = x_res / x_step + 1;
  float step_size = x_step * _res;

  for (const auto &pair : submaps_) {
    static int map_num = 0;
    cout << "Meshing submap " << ++map_num << "\n";
    freeze_all_maps();
    pair.second->activate();
    auto pos = pair.second->t_pos_W_M_;

    for (int i = 0; i < steps; ++i) {
      auto start = i * step_size + k_x_min;
      auto end = start + step_size;
      if (i == steps - 1) {
        end = end > k_x_max ? k_x_max : end;
      }
      if (end == start)
        continue;

      auto mesh_xyz =
          utils::meshgrid_3d(start, end + _res, k_y_min, k_y_max + _res,
                             k_z_min, k_z_max + _res, _res, *p_device) +
          pos.view({1, 1, 1, 3});

      auto x_num = mesh_xyz.size(0);
      auto y_num = mesh_xyz.size(1);
      auto z_num = mesh_xyz.size(2);
      mesh_xyz = mesh_xyz.view({-1, 3});

      auto mesh_mask = get_conf_mask(mesh_xyz, k_layer_num - 1);
      auto mask_xyz_sdf = get_sdf(mesh_xyz.index({mesh_mask}));
      auto mesh_sdf = torch::zeros({mesh_xyz.size(0), 1}).to(*p_device);
      mesh_sdf.index_put_({mesh_mask}, mask_xyz_sdf);

      /// [n,3,3]
      auto face_xyz = xyz_sdf_mask_to_face(mesh_xyz, mesh_sdf, mesh_mask, x_num,
                                           y_num, z_num);
      if (face_xyz.size(0) == 0)
        continue;

      pub_mesh(_mesh_pub, _mesh_color_pub, face_xyz, _header, _uuid);
      p_mesher_->vec_face_xyz_.emplace_back(face_xyz.cpu());
    }
    if (p_mesher_->vec_face_xyz_.empty())
      continue;

    if (k_large_scene) {
      auto face_xyz = torch::cat({p_mesher_->vec_face_xyz_}, 0);
      std::string filename =
          k_output_path + "/mesh_" + to_string(map_num) + ".ply";
      face_to_ply(face_xyz, filename);
      p_mesher_->vec_face_xyz_.clear();
    }
  }
  cout << "End of global map meshing."
       << "\n";
}

void GlobalMap::voxeling(ros::Publisher &voxel_map_pub,
                         std_msgs::Header &_header) {
  if (voxel_map_pub.getNumSubscribers() > 0) {

    cout << "Start meshing global map..."
         << "\n";
    auto vis_mesh_xyz_coarse =
        utils::meshgrid_3d(k_x_min, k_x_max, k_y_min, k_y_max, k_z_min, k_z_max,
                           voxel_sizes[k_layer_num - 1], *p_device)
            .view({-1, 3});
    auto vis_mesh_xyz_fine =
        utils::meshgrid_3d(0, voxel_sizes[k_layer_num - 1], 0,
                           voxel_sizes[k_layer_num - 1], 0,
                           voxel_sizes[k_layer_num - 1], k_vis_res, *p_device)
            .view({-1, 3});
    for (const auto &submap : submaps_) {
      auto submap_index = submap.first;
      auto pos = submap.second->t_pos_W_M_;

      // voxelized_xyz
      torch::Tensor tmp_vis_mesh_xyz_coarse;
      tmp_vis_mesh_xyz_coarse = vis_mesh_xyz_coarse + pos;

      if (k_skip_unconf) {
        tmp_vis_mesh_xyz_coarse =
            get_conf_points(tmp_vis_mesh_xyz_coarse, k_layer_num - 1);
      }

      auto xyz_voxelized = (tmp_vis_mesh_xyz_coarse.unsqueeze(1) +
                            vis_mesh_xyz_fine.unsqueeze(0))
                               .contiguous()
                               .view({-1, 3});
      if (xyz_voxelized.size(0) == 0) {
        continue;
      }
      auto vis_voxel_map = utils::get_vix_voxel_map(
          torch::cat(xyz_voxelized, 0), voxel_sizes[0], 1.0, 0.0, 0.0);
      if (!vis_voxel_map.points.empty()) {
        vis_voxel_map.header = _header;
        voxel_map_pub.publish(vis_voxel_map);
      }
    }
    cout << "End of global map meshing."
         << "\n";
  }
}

void GlobalMap::main_loop() {
  // TicToc tic;
  while (true) {
    std::unique_lock<std::mutex> lock(buf_mutex);
    buf_cond.wait(lock);
    while (!save_buf.empty()) {
      auto [xyzs, feats, confs, layer] = save_buf.front();
      save_buf.pop();
      lock.unlock();
      save_(xyzs, feats, confs, layer);
      lock.lock();
    }
    lock.unlock();
  }
}

void GlobalMap::save_(const torch::Tensor &_w_index,
                      const torch::Tensor &_feats, const torch::Tensor &_confs,
                      int _layer) {
  auto submap_indices =
      (((_w_index + offsets_index[_layer]) / submap_size_index_[_layer]))
          .floor()
          .to(torch::kLong);
  auto submap_indices_uniq = (get<0>(torch::unique_dim(submap_indices, 0)));
  for (int i = 0; i < submap_indices_uniq.size(0); ++i) {
    auto submap_index = submap_indices_uniq[i];
    auto it = submaps_.find(submap_index);
    SubMap *p_submap;
    if (it == submaps_.end()) {
      auto submap_xyz = submap_index * submap_size_;
      p_submap = new SubMap(submap_xyz, k_x_min, k_x_max, k_y_min, k_y_max,
                            k_z_min, k_z_max);
      submaps_[submap_index] = p_submap;
      ++active_map_num_;
    } else {
      p_submap = it->second;
      if (!p_submap->active_) {
        p_submap->activate();
        ++active_map_num_;
      }
    }
    latest_submap_index_ = submap_index;

    auto mask = submap_indices == submap_index.unsqueeze(0);
    mask = mask.all(1);

    auto voxel_index = _w_index.index({mask});
    auto submap_idx = p_submap->world_index_to_idx(voxel_index, _layer);

    p_submap->feat_maps_[_layer].data().index_put_({submap_idx},
                                                   _feats.data().index({mask}));
    p_submap->conf_maps_[_layer].index_put_({submap_idx}, _confs.index({mask}));
  }
}

void GlobalMap::freeze_old_maps() {
  for (const auto &submap : submaps_) {
    if (submap.second->active_) {
      if ((latest_submap_index_ - submap.first)
              .to(torch::kFloat32)
              .norm()
              .item<float>() > 2) {
        submap.second->freeze();
        --active_map_num_;
      }
    }
  }
}

void GlobalMap::freeze_all_maps() {
  for (const auto &pair : submaps_) {
    if (pair.second->active_) {
      pair.second->freeze();
      --active_map_num_;
    }
  }
}

torch::Tensor GlobalMap::get_conf_mask(const torch::Tensor &_xyzs, int _layer) {
  return fetch_attribute(_xyzs, _layer, 1) > 0;
}

torch::Tensor GlobalMap::get_conf_mask_index(const torch::Tensor &_w_index,
                                             int _layer) {
  return fetch_attribute_index(_w_index, _layer, 1) > 0;
}

torch::Tensor GlobalMap::get_conf_points(const torch::Tensor &_xyzs,
                                         int _layer) {
  return _xyzs.index({get_conf_mask(_xyzs, _layer)});
}

torch::Tensor GlobalMap::fetch_attribute(const torch::Tensor &_xyzs, int _layer,
                                         int _type) {
  auto submap_indices =
      ((_xyzs + offsets_) / submap_size_).floor().to(torch::kLong);
  auto submap_indices_uniq = (get<0>(torch::unique_dim(submap_indices, 0)));
  torch::Tensor attribute;

  if (_type == 0) {
    if (k_zero_init) {
      attribute = torch::zeros({_xyzs.size(0), k_feat_dim}, *p_device);
    } else {
      attribute = torch::randn({_xyzs.size(0), k_feat_dim}, *p_device);
    }
  } else {
    attribute = torch::zeros({_xyzs.size(0)}, *p_device).to(torch::kInt);
  }
  // #pragma omp parallel for reduction(+ : active_map_num_)
  for (int i = 0; i < submap_indices_uniq.size(0); ++i) {
    auto submap_index = submap_indices_uniq[i];
    auto it = submaps_.find(submap_index);
    if (it == submaps_.end()) {
      continue;
    } else if (!it->second->active_) {
      it->second->activate();
      ++active_map_num_;
    }

    auto mask = submap_indices == submap_index.unsqueeze(0);
    mask = mask.all(1);

    auto voxel_xyzs = _xyzs.index({mask});
    auto submap_idx = it->second->xyz_to_idx(voxel_xyzs, _layer);

    if (_type == 0) {
      attribute.data().index_put_(
          {mask}, it->second->feat_maps_[_layer].data().index({submap_idx}));
    } else {
      attribute.index_put_(
          {mask}, it->second->conf_maps_[_layer].index({submap_idx}).cuda());
    }
  }
  return attribute;
}

torch::Tensor GlobalMap::fetch_attribute_index(const torch::Tensor &_w_index,
                                               int _layer, int _type) {
  auto submap_indices =
      (((_w_index + offsets_index[_layer]) / submap_size_index_[_layer]))
          .floor()
          .to(torch::kLong);
  auto submap_indices_uniq = (get<0>(torch::unique_dim(submap_indices, 0)));
  torch::Tensor attribute;
  if (_type == 0) {
    attribute = torch::zeros({_w_index.size(0), k_feat_dim}, *p_device);
  } else {
    attribute = torch::zeros({_w_index.size(0)}, *p_device).to(torch::kInt);
  }
  // #pragma omp parallel for reduction(+ : active_map_num_)
  for (int i = 0; i < submap_indices_uniq.size(0); ++i) {
    auto submap_index = submap_indices_uniq[i];
    auto it = submaps_.find(submap_index);
    if (it == submaps_.end()) {
      continue;
    } else if (!it->second->active_) {
      it->second->activate();
      ++active_map_num_;
    }

    auto mask = submap_indices == submap_index.unsqueeze(0);
    mask = mask.all(1);

    auto voxel_xyzs = _w_index.index({mask});
    auto submap_idx = it->second->world_index_to_idx(voxel_xyzs, _layer);

    if (_type == 0) {
      attribute.data().index_put_(
          {mask}, it->second->feat_maps_[_layer].data().index({submap_idx}));
    } else {
      attribute.index_put_({mask},
                           it->second->conf_maps_[_layer].index({submap_idx}));
    }
  }
  return attribute;
}
