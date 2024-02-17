#include "neural_slam.h"

#include "marching_cube/marching_cube.h"
#include "mesh_msgs/MeshVertexColorsStamped.h"

#include <cv_bridge/cv_bridge.h>
#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>
#include <tf/transform_broadcaster.h>

#include "llog/llog.h"
#include "params/params.h"

using namespace std;

NeuralSLAM::NeuralSLAM(ros::NodeHandle &_nh, const std::string &_config_path,
                       const std::string &_data_path) {
  cout << "TORCH_VERSION: " << TORCH_VERSION << '\n';

  read_params(_config_path, _data_path);

  p_local_map = std::make_unique<LocalMap>();
  p_global_map = std::make_unique<GlobalMap>();

  for (auto &p : p_local_map->named_parameters()) {
    cout << p.key() << p.value().sizes() << '\n';
  }
  p_sdf_optimizer =
      std::make_unique<torch::optim::Adam>(p_local_map->parameters(), k_lr);

  TRIANGLE_TABLE = TRIANGLE_TABLE.to(*p_device);
  EDGE_INDEX_PAIRS = EDGE_INDEX_PAIRS.to(*p_device);

  if (k_online) {
    register_subscriber(_nh);
  } else {
    p_data_loader =
        std::make_unique<DataLoader>(k_calib_file, k_pose_file, k_depth_path);
  }
  register_publisher(_nh);

  mapper_thread = std::thread(&NeuralSLAM::mapper_loop, this);
  keyboard_thread = std::thread(&NeuralSLAM::keyboard_loop, this);
}

void NeuralSLAM::register_subscriber(ros::NodeHandle &nh) {
  if (k_depth_type == 0) {
    depth_sub = nh.subscribe(
        k_depth_topic, 100,
        &NeuralSLAM::depth_callback<sensor_msgs::PointCloud2ConstPtr>, this);
  } else if (k_depth_type == 1) {
    depth_sub = nh.subscribe(
        k_depth_topic, 100,
        &NeuralSLAM::depth_callback<sensor_msgs::ImageConstPtr>, this);
  }

  if (k_pose_msg_type == 0) {
    pose_sub = nh.subscribe(
        k_pose_topic, 100,
        &NeuralSLAM::pose_callback<nav_msgs::OdometryConstPtr>, this);
  } else if (k_pose_msg_type == 1) {
    pose_sub = nh.subscribe(
        k_pose_topic, 100,
        &NeuralSLAM::pose_callback<geometry_msgs::PoseStampedConstPtr>, this);
  } else if (k_pose_msg_type == 2) {
    pose_sub = nh.subscribe(
        k_pose_topic, 100,
        &NeuralSLAM::pose_callback<geometry_msgs::TransformStampedConstPtr>,
        this);
  }
}

void NeuralSLAM::register_publisher(ros::NodeHandle &nh) {
  pose_pub = nh.advertise<geometry_msgs::PoseStamped>("pose", 1);
  path_pub = nh.advertise<nav_msgs::Path>("path", 1);
  loss_pub = nh.advertise<std_msgs::Float32>("loss", 1);
  mesh_pub = nh.advertise<mesh_msgs::MeshGeometryStamped>("mesh", 1);
  mesh_color_pub =
      nh.advertise<mesh_msgs::MeshVertexColorsStamped>("mesh_color", 1);
  global_mesh_pub =
      nh.advertise<mesh_msgs::MeshGeometryStamped>("global_mesh", 1);
  global_mesh_color_pub =
      nh.advertise<mesh_msgs::MeshVertexColorsStamped>("global_mesh_color", 1);
  tsdf_mesh_pub = nh.advertise<mesh_msgs::MeshGeometryStamped>("tsdf_mesh", 1);
  voxel_pub = nh.advertise<visualization_msgs::Marker>("voxel", 1);
  global_voxel_pub =
      nh.advertise<visualization_msgs::Marker>("global_voxel", 1);
  sdf_map_pub = nh.advertise<visualization_msgs::Marker>("sdf_map", 1);
  vis_shift_map_pub =
      nh.advertise<visualization_msgs::Marker>("vis_shift_map", 1);
}

template <typename DepthMsgT>
void NeuralSLAM::depth_callback(const DepthMsgT &_depth_msg) {
  // static double last_frame_time = 0;
  // if ((_depth_msg->header.stamp.toSec() - last_frame_time) < inv_frame_rate)
  // {
  //   return;
  // }
  // last_frame_time = _depth_msg->header.stamp.toSec();

  torch::Tensor rays_d, depths;
  if constexpr (std::is_same<DepthMsgT,
                             sensor_msgs::PointCloud2ConstPtr>::value) {
    PointCloudT pcl_cloud;
    pcl::fromROSMsg(*_depth_msg, pcl_cloud);
    utils::pointcloud_to_raydepth(pcl_cloud, k_ds_pt_num, rays_d, depths,
                                  k_min_range);
  } else if constexpr (std::is_same<DepthMsgT,
                                    sensor_msgs::ImageConstPtr>::value) {
    cv::Mat depth_img = cv_bridge::toCvShare(_depth_msg)->image;
    if (_depth_msg->encoding == "mono16" || _depth_msg->encoding == "16UC1") {
      depth_img.convertTo(depth_img, CV_32FC1, 1e-3);
    }
    depths =
        torch::from_blob(depth_img.data, {depth_img.rows, depth_img.cols, 1},
                         torch::kFloat32)
            .view({-1, 1});

    // filter out nan
    auto mask = (depths > 0).squeeze();
    depths = depths.index({mask});
    if (depths.size(0) < 1)
      return;

    depths = utils::downsample_point(depths, k_ds_pt_num);

    auto vu = utils::meshgrid_2d(depth_img.cols, depth_img.rows);
    vu = vu.index({mask});
    vu = utils::downsample_point(vu, k_ds_pt_num);

    auto pt_x = (vu.select(1, 1) - k_cx) / k_fx;
    auto pt_y = (vu.select(1, 0) - k_cy) / k_fy;
    auto pt_z = torch::ones_like(pt_x);
    rays_d = torch::stack({pt_x, pt_y, pt_z}, -1);
  }
  mapper_buf_mutex.lock();
  mapper_header_buf.emplace(_depth_msg->header);
  mapper_pcl_buf.emplace(torch::cat({rays_d, depths}, 1));
  mapper_buf_mutex.unlock();
}

template <typename PoseMsgT>
void NeuralSLAM::pose_callback(const PoseMsgT &pose_msg) {
  torch::Tensor pos, quat;
  if constexpr (std::is_same<PoseMsgT, nav_msgs::OdometryConstPtr>::value) {
    pos = torch::tensor({pose_msg->pose.pose.position.x,
                         pose_msg->pose.pose.position.y,
                         pose_msg->pose.pose.position.z},
                        torch::kFloat);
    quat = torch::tensor(
        {pose_msg->pose.pose.orientation.w, pose_msg->pose.pose.orientation.x,
         pose_msg->pose.pose.orientation.y, pose_msg->pose.pose.orientation.z},
        torch::kFloat);
  } else if constexpr (std::is_same<
                           PoseMsgT,
                           geometry_msgs::PoseStampedConstPtr>::value) {
    pos = torch::tensor({pose_msg->pose.position.x, pose_msg->pose.position.y,
                         pose_msg->pose.position.z},
                        torch::kFloat);
    quat = torch::tensor(
        {pose_msg->pose.orientation.w, pose_msg->pose.orientation.x,
         pose_msg->pose.orientation.y, pose_msg->pose.orientation.z},
        torch::kFloat);
  } else if constexpr (std::is_same<
                           PoseMsgT,
                           geometry_msgs::TransformStampedConstPtr>::value) {
    pos = torch::tensor({pose_msg->transform.translation.x,
                         pose_msg->transform.translation.y,
                         pose_msg->transform.translation.z},
                        torch::kFloat);
    quat = torch::tensor(
        {pose_msg->transform.rotation.w, pose_msg->transform.rotation.x,
         pose_msg->transform.rotation.y, pose_msg->transform.rotation.z},
        torch::kFloat);
  }
  auto rot = utils::quat_to_rot(quat);
  auto rot_pos = torch::cat({rot.mm(k_T_B_S.slice(1, 0, 3)),
                             rot.mm(k_T_B_S.slice(1, 3, 4)) + pos.unsqueeze(1)},
                            1);
  mapper_buf_mutex.lock();
  pose_msg_buf.emplace(pose_msg->header, rot_pos);
  mapper_buf_mutex.unlock();
  mapperer_buf_cond.notify_one();
}

torch::Tensor NeuralSLAM::train(const torch::Tensor &_pos,
                                const torch::Tensor &_xyz_rays_depths,
                                int _opt_iter) {
  torch::GradMode::set_enabled(false);
  if (!is_init_ || k_shift_map) {
    static auto p_t_move = llog::CreateTimer(" move");
    p_t_move->tic();
    p_local_map->move_to(_pos, *p_global_map);
    is_init_ = true;
    p_t_move->toc_sum();
  }

  static auto p_t_his_pt = llog::CreateTimer(" hist_pt");
  p_t_his_pt->tic();
  // ir: inrange
  auto ir_mask = p_local_map->get_inrange_mask(_xyz_rays_depths.slice(1, 0, 3));
  auto ir_xyz_rays_depths = _xyz_rays_depths.index({ir_mask});

  // or: outrange
  torch::Tensor or_xyz_rays_zfars_ddepths;
  if (k_use_outrange) {
    or_xyz_rays_zfars_ddepths = _xyz_rays_depths.index({~ir_mask});

    torch::Tensor z_nears, z_fars, mask_intersect;
    p_local_map->get_intersect_point(or_xyz_rays_zfars_ddepths.slice(1, 0, 3),
                                     or_xyz_rays_zfars_ddepths.slice(1, 3, 6),
                                     z_nears, z_fars, mask_intersect);
    // [n,8]
    or_xyz_rays_zfars_ddepths = torch::cat(
        {or_xyz_rays_zfars_ddepths.slice(1, 0, 6),
         or_xyz_rays_zfars_ddepths.slice(1, 6, 7) + z_fars.unsqueeze(1),
         -z_fars.unsqueeze(1)},
        1);
    or_xyz_rays_zfars_ddepths =
        or_xyz_rays_zfars_ddepths.index({mask_intersect});
  }

  static torch::Tensor his_pts_rays_depths;
  if (his_pts_rays_depths.size(0) > k_max_pt_num) {
    his_pts_rays_depths =
        utils::downsample_point(his_pts_rays_depths, k_max_pt_num);
  }

  if (his_pts_rays_depths.size(0) == 0 || !k_hist_pt) {
    his_pts_rays_depths = ir_xyz_rays_depths;
  } else {
    auto mask =
        p_local_map->get_inrange_mask(his_pts_rays_depths.slice(1, 0, 3));
    his_pts_rays_depths = his_pts_rays_depths.index({mask});

    his_pts_rays_depths =
        torch::cat({his_pts_rays_depths, ir_xyz_rays_depths}, 0);
  }

  p_t_his_pt->toc_avg();

  RaySamples samples;
  // std_msgs::Float32 loss_ros;
  for (int i = 0; i < _opt_iter; ++i) {
    static auto p_t_sample = llog::CreateTimer("  sample");
    p_t_sample->tic();
    torch::Tensor batch_pts_rays_depths;
    utils::sample_batch_pts(his_pts_rays_depths, batch_pts_rays_depths,
                            k_batch_pt_num, k_batch_type, i);

    // [n,num_samples]
    utils::sample(batch_pts_rays_depths, samples, k_surface_sample_num,
                  k_free_sample_num, k_sample_std, k_strat_near_ratio,
                  k_strat_far_ratio, k_dist_type);

    if (k_use_outrange && or_xyz_rays_zfars_ddepths.size(0) > 0) {
      torch::Tensor batch_or_pts_rays_zfars_ddepths;
      utils::sample_batch_pts(or_xyz_rays_zfars_ddepths,
                              batch_or_pts_rays_zfars_ddepths, k_batch_pt_num,
                              k_batch_type, i);

      // [n,num_samples]
      torch::Tensor sample_or_pts, sample_or_dirs, sample_or_gts;
      utils::sample_strat_pts(batch_or_pts_rays_zfars_ddepths, sample_or_pts,
                              sample_or_dirs, sample_or_gts, k_out_sample_num,
                              k_sample_std, 0.1, 0.9);
      sample_or_gts +=
          batch_or_pts_rays_zfars_ddepths.slice(1, 7, 8).unsqueeze(1);
      if (k_dist_type) {
        auto dist = utils::cal_nn_dist(
            sample_or_pts, batch_or_pts_rays_zfars_ddepths.slice(1, 0, 3));
        sample_or_gts = sample_or_gts.view({-1});
        auto mask = dist < sample_or_gts;
        sample_or_gts.index_put_({mask}, dist.index({mask}));
      }

      samples.xyz = torch::cat({samples.xyz, sample_or_pts.view({-1, 3})}, 0);
      samples.ray_sdf =
          torch::cat({samples.ray_sdf, sample_or_gts.view({-1, 1})}, 0);
      samples.direction =
          torch::cat({samples.direction, sample_or_dirs.view({-1, 3})}, 0);
    }

    auto mask = p_local_map->get_inrange_mask(samples.xyz);
    samples.xyz = samples.xyz.index({mask});
    samples.ray_sdf = samples.ray_sdf.index({mask});
    samples.direction = samples.direction.index({mask});
    if (samples.xyz.size(0) < 1)
      continue;

    torch::GradMode::set_enabled(true);
    if (k_eikonal) {
      samples.xyz.requires_grad_(true);
    }
    p_t_sample->toc_avg();

    samples.pred_sdf = p_local_map->get_sdf(samples.xyz);

    if (k_eikonal) {
      static auto p_t_grad = llog::CreateTimer("  gradient");
      p_t_grad->tic();
      auto d_output = torch::ones_like(samples.pred_sdf);
      samples.pred_gradient = torch::autograd::grad(
          {samples.pred_sdf}, {samples.xyz}, {d_output}, true, true)[0];
      p_t_grad->toc_avg();
    }

    static auto p_t_backward = llog::CreateTimer("  backward");
    p_t_backward->tic();
    torch::Tensor loss;
    if (k_supervise_mode == 0) {
      loss = torch::l1_loss(samples.pred_sdf, samples.ray_sdf);
    } else {
      auto bce_gts = 1 / (1 + torch::exp(samples.ray_sdf / k_bce_sigma));
      auto xyz_pred_bce_sdf =
          1 / (1 + torch::exp(samples.pred_sdf / k_bce_sigma));
      loss = torch::binary_cross_entropy(xyz_pred_bce_sdf, bce_gts);
    }

    if (k_eikonal) {
      auto mask = (samples.ray_sdf > k_truncated_dis).squeeze();
      if (mask.any().item<bool>()) {
        loss += k_eikonal_weight *
                (samples.pred_gradient.index({mask}).norm(2, 1) - 1.0)
                    .square()
                    .mean();
      }
    }
    p_sdf_optimizer->zero_grad();
    loss.backward();
    p_sdf_optimizer->step();
    p_t_backward->toc_avg();

    // printf("iter: %d, loss: %f", i + 1, loss.item<float>());
    // loss_ros.data = loss.item<float>();
    // loss_pub.publish(loss_ros);
  }

  // remove points in his_pts_rays_depths that theirs bce_loss is over 0.5
  if (k_outlier_remove) {
    static int count = 0;
    if (++count % k_frame_rate == 0) {
      static auto p_t_outlier_remove = llog::CreateTimer(" outlier_remove");
      p_t_outlier_remove->tic();
      auto xyz_pred_sdf =
          p_local_map->get_sdf(his_pts_rays_depths.slice(1, 0, 3));

      auto mask = (xyz_pred_sdf.abs() < k_outlier_dist).squeeze();
      cout << "before:" << his_pts_rays_depths.sizes() << '\n';
      his_pts_rays_depths = his_pts_rays_depths.index({mask});
      cout << "after:" << his_pts_rays_depths.sizes() << '\n';
      p_t_outlier_remove->toc_sum();
    }
  }

  return samples.xyz;
}

bool NeuralSLAM::get_input(torch::Tensor &_pose, std_msgs::Header &_header,
                           torch::Tensor &_xyz_rays_depths) {
  static auto p_timer = llog::CreateTimer("get_input");
  p_timer->tic();
  if (k_online) {
    if (!pose_msg_buf.empty() && !mapper_pcl_buf.empty()) {

      bool is_sync_data = false;
      mapper_buf_mutex.lock();
      _header = mapper_header_buf.front();
      auto rays_depths = mapper_pcl_buf.front().to(*p_device);
      mapper_header_buf.pop();
      mapper_pcl_buf.pop();

      while (!pose_msg_buf.empty() && pose_msg_buf.front().first.stamp <
                                          _header.stamp + ros::Duration(0.01)) {
        _pose = pose_msg_buf.front().second.to(*p_device);
        _header.frame_id = pose_msg_buf.front().first.frame_id;
        pose_msg_buf.pop();
        is_sync_data = true;
      }
      mapper_buf_mutex.unlock();
      if (!is_sync_data) {
        p_timer->toc_sum();
        return false;
      }

      auto rot = _pose.slice(1, 0, 3);
      auto pos = _pose.slice(1, 3, 4).view({1, 3});

      auto _tmp_rays_d = rays_depths.slice(1, 0, 3);
      auto _tmp_depths = rays_depths.slice(1, 3, 4);
      //[n,3]
      _tmp_rays_d = _tmp_rays_d.mm(rot.t());
      //[n,3]
      auto _tmp_xyz = _tmp_rays_d * _tmp_depths + pos;
      // [n,7]
      _xyz_rays_depths = torch::cat({_tmp_xyz, _tmp_rays_d, _tmp_depths}, 1);
      p_timer->toc_sum();
      return true;
    } else {
      return false;
    }
  } else {
    _header.frame_id = "world";
    _header.stamp = ros::Time::now();

    static int cur_idx = 0;
    if (cur_idx == 0 || (cur_idx + 1) % k_every_frame == 0) {
      torch::Tensor points, colors;
      if (p_data_loader->get_next_data(cur_idx, _pose, points, colors,
                                       *p_device)) {
        points = utils::downsample_point(points, k_ds_pt_num);
        // colors = utils::downsample_point(colors, k_ds_pt_num);

        // [n,3],[n,3],[n,1]
        auto depths = points.norm(2, 1, true);
        auto rays_d = points.div(depths);

        // filter out nan
        auto valid_idx = (depths.squeeze() > k_min_range).nonzero().squeeze();
        depths = depths.index({valid_idx});
        rays_d = rays_d.index({valid_idx});

        _pose = _pose.to(*p_device);

        auto rot = _pose.slice(1, 0, 3);
        auto pos = _pose.slice(1, 3, 4).squeeze();
        //[n,3]
        rays_d = rays_d.mm(rot.t());
        //[n,3]
        auto _tmp_xyz = rays_d * depths + pos;
        // [n,7]
        _xyz_rays_depths = torch::cat({_tmp_xyz, rays_d, depths}, 1);
        cur_idx++;
        p_timer->toc_sum();
        return true;
      } else {
        return end();
      }
    } else {
      cur_idx++;
      return false;
    }
  }
}

void NeuralSLAM::mapper_loop() {
  int frame_num = 0;
  torch::Tensor pose, xyz_rays_d_depths;
  std_msgs::Header header;

  static ofstream mem_file(k_output_path + "/mem_usage.txt");
  mem_file << "frame_num\tcpu_mem_usage\tgpu_mem_usage\ttiming\n";
  while (true) {
    if (get_input(pose, header, xyz_rays_d_depths)) {
      static auto p_train = llog::CreateTimer("train");
      p_train->tic();

      if (k_decoder_freeze_frame_num > 0 &&
          frame_num > k_decoder_freeze_frame_num && k_shift_map) {
        p_local_map->freeze_decoder();
      }
      int iter_step = k_iter_step;
      auto pos = pose.slice(1, 3, 4).squeeze();
      torch::Tensor xyz_sample = train(pos, xyz_rays_d_depths, iter_step);

      p_global_map->save_decoder(p_local_map->named_parameters());
      if (p_global_map->active_map_num_ > 10) {
        p_global_map->freeze_old_maps();
      }

      p_train->toc_sum();

      mapper_init = true;
      mapper_update = true;

      if ((frame_num + 1) % k_vis_frame_step == 0) {
        visualization(pose, xyz_rays_d_depths.slice(1, 0, 3), header);
      }

      // save gt
      /* auto inrange_xyz = xyz_rays_d_depths.slice(1, 0, 3).index(
          {p_local_map->get_inrange_mask(xyz_rays_d_depths.slice(1, 0, 3))});
      vec_inrange_xyz.emplace_back(inrange_xyz.cpu());
      auto rot = pose.slice(1, 0, 3);
      auto xyz_local = (inrange_xyz - pos).mm(rot);
      // convert frame_num to string with 6 digits
      export_to_ply(xyz_local.cpu(), k_output_path,
                    cv::format("%06d", frame_num)); */

      // save current cpu and gpu memory usage to file, it extremely slow
      mem_file << frame_num << "\t" << utils::get_cpu_mem_usage() << "\t"
               << utils::get_gpu_mem_usage() << "\t" << p_train->this_time()
               << '\n';
      ++frame_num;

      llog::PrintLog();
    } else {
      std::chrono::milliseconds dura(20);
      std::this_thread::sleep_for(dura);
    }
  }
}

void NeuralSLAM::pub_pose(const torch::Tensor &_pose,
                          const std_msgs::Header &_header) {

  auto pos = _pose.slice(1, 3, 4).detach().cpu();
  auto quat = utils::rot_to_quat(_pose.slice(1, 0, 3).detach().cpu());
  geometry_msgs::PoseStamped pose_stamped;
  pose_stamped.header = _header;
  pose_stamped.pose.position.x = pos[0].item<float>();
  pose_stamped.pose.position.y = pos[1].item<float>();
  pose_stamped.pose.position.z = pos[2].item<float>();
  pose_stamped.pose.orientation.w = quat[0].item<float>();
  pose_stamped.pose.orientation.x = quat[1].item<float>();
  pose_stamped.pose.orientation.y = quat[2].item<float>();
  pose_stamped.pose.orientation.z = quat[3].item<float>();
  pose_pub.publish(pose_stamped);

  path_msg.header = _header;
  path_msg.poses.emplace_back(pose_stamped);
  path_pub.publish(path_msg);

  static tf::TransformBroadcaster br;
  tf::Transform transform;
  tf::Quaternion q;
  // body frame
  transform.setOrigin(tf::Vector3(pose_stamped.pose.position.x,
                                  pose_stamped.pose.position.y,
                                  pose_stamped.pose.position.z));
  q.setW(pose_stamped.pose.orientation.w);
  q.setX(pose_stamped.pose.orientation.x);
  q.setY(pose_stamped.pose.orientation.y);
  q.setZ(pose_stamped.pose.orientation.z);
  transform.setRotation(q);
  br.sendTransform(
      tf::StampedTransform(transform, _header.stamp, "world", "depth"));
  transform.setIdentity();
  br.sendTransform(tf::StampedTransform(transform, _header.stamp, "world",
                                        _header.frame_id));
}

void NeuralSLAM::visualization(const torch::Tensor &_pose,
                               const torch::Tensor &_xyz,
                               const std_msgs::Header &_header) {
  static auto p_timer = llog::CreateTimer("visualization");
  p_timer->tic();
  torch::GradMode::set_enabled(false);

  if (pose_pub.getNumSubscribers() > 0 || path_pub.getNumSubscribers() > 0) {
    pub_pose(_pose, _header);
  }

  auto tmp_header = _header;
  tmp_header.frame_id = "world";

  if (mesh_pub.getNumSubscribers() > 0) {
    if (k_eval_mode > -1) {
      p_local_map->meshing_(mesh_pub, mesh_color_pub, tmp_header, k_vis_res,
                            false, _xyz);
    }
  }
  if (voxel_pub.getNumSubscribers() > 0) {
    auto vis_voxel_map = utils::get_vix_voxel_map(_xyz, voxel_sizes[0]);
    if (!vis_voxel_map.points.empty()) {
      vis_voxel_map.header = tmp_header;
      voxel_pub.publish(vis_voxel_map);
    }
  }
  if (sdf_map_pub.getNumSubscribers() > 0) {
    auto vis_sdf_map =
        p_local_map->get_sdf_map_(p_local_map->t_pos_W_M_, _xyz, k_vis_res,
                                  k_slice_height, 2 * k_truncated_dis);
    if (!vis_sdf_map.points.empty()) {
      vis_sdf_map.header = tmp_header;
      sdf_map_pub.publish(vis_sdf_map);
    }
  }
  if (vis_shift_map_pub.getNumSubscribers() > 0) {
    auto vis_shift_map =
        utils::get_vis_shift_map(p_local_map->t_pos_W_M_, k_x_min, k_x_max,
                                 k_y_min, k_y_max, k_z_min, k_z_max);
    if (!vis_shift_map.points.empty()) {
      vis_shift_map.header = tmp_header;
      vis_shift_map_pub.publish(vis_shift_map);
    }
  }
  p_timer->toc_sum();
}

void NeuralSLAM::keyboard_loop() {
  while (true) {
    char c = getchar();
    if (c == 's') {
      save_mesh();
    } else if (c == 'v') {
      p_local_map->save_to_globalmap(*p_global_map);
      std_msgs::Header header;
      header.frame_id = "world";
      header.stamp = ros::Time::now();
      p_global_map->meshing(global_mesh_pub, global_mesh_color_pub, header,
                            k_export_res);
      p_global_map->voxeling(global_voxel_pub, header);
    } else if (c == 'g') {
      auto pcl = torch::cat({vec_inrange_xyz}, 0);
      utils::export_to_ply(pcl, k_output_path);
    } else if (c == 'e') {
      eval_mesh();
    }

    std::chrono::milliseconds dura(100);
    std::this_thread::sleep_for(dura);
  }
}

void NeuralSLAM::export_checkpoint() {
  if (!k_output_path.empty()) {
    torch::save(p_local_map->parameters(),
                k_output_path + "/local_map_checkpoint.pt");

    // TODO: save global map
    // for (int i = 0; i < p_global_map->active_map_num_; ++i) {
    //   torch::save(p_global_map->submaps_[i]->named_parameters(),
    //               k_output_path + "/global_map_checkpoint_" +
    //                   std::to_string(i) + ".pt");
    // }
  }
}

void NeuralSLAM::load_checkpoint(const std::string &_checkpoint_path) {
  // torch::load(p_local_map->parameters(), _checkpoint_path);
}

void NeuralSLAM::save_mesh() {
  if (k_shift_map) {
    if (p_global_map->p_mesher_->vec_face_xyz_.empty()) {
      p_local_map->save_to_globalmap(*p_global_map);
      std_msgs::Header header;
      header.frame_id = "world";
      header.stamp = ros::Time::now();
      p_global_map->meshing(global_mesh_pub, global_mesh_color_pub, header,
                            k_export_res);
    }
    if (!k_large_scene)
      p_global_map->p_mesher_->save_mesh();
  } else {
    std_msgs::Header header;
    header.frame_id = "world";
    header.stamp = ros::Time::now();
    p_local_map->meshing_(mesh_pub, mesh_color_pub, header, k_export_res, true);
    p_local_map->p_mesher_->save_mesh();
  }
}

void NeuralSLAM::eval_mesh() {
  auto eval_python_cmd = "python src/RIM/rim/eval/evaluator.py --pred_mesh " +
                         k_output_path + "/mesh.ply --gt_pcd " +
                         k_gt_structure_file;
  printf("\033[1;34mConducting evaluation command: %s\n\033[0m",
         eval_python_cmd.c_str());
  int ret = std::system(eval_python_cmd.c_str());
  printf("\033[1;32mEvaluation finished.\n"
         "Please check the results in the folder: %s\n\033[0m",
         (k_output_path).c_str());
}

void NeuralSLAM::export_timing() {
  llog::SaveLog(k_output_path + "/timing.txt");
}

/**
 * @description: End the program and save the outputs.
 * @return {*}
 */
bool NeuralSLAM::end() {
  export_timing();
  export_checkpoint();
  save_mesh();
  eval_mesh();

  exit(0);
  return true;
}