#pragma once

#include <Eigen/Core>
#include <ros/ros.h>

#include "data_loader.hpp"
#include "neural_net/local_map.h"
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/TransformStamped.h>

class NeuralSLAM {
public:
  NeuralSLAM(ros::NodeHandle &nh, const std::string &_config_path,
             const std::string &_data_path = "");

private:
  std::vector<torch::Tensor> vec_inrange_xyz;
  std::unique_ptr<LocalMap> p_local_map;
  std::unique_ptr<GlobalMap> p_global_map;
  std::unique_ptr<torch::optim::Adam> p_sdf_optimizer;

  std::unique_ptr<DataLoader> p_data_loader;

  // ROS stuff
  ros::Subscriber depth_sub, pose_sub;
  ros::Publisher pose_pub, path_pub;
  ros::Publisher loss_pub, mesh_pub, mesh_color_pub, global_mesh_pub,
      global_mesh_color_pub, tsdf_mesh_pub, voxel_pub, global_voxel_pub,
      sdf_map_pub, vis_shift_map_pub;

  nav_msgs::Path path_msg;

  // thread, buffer stuff
  std::mutex mapper_buf_mutex;
  std::condition_variable mapperer_buf_cond;
  std::queue<torch::Tensor> mapper_pcl_buf, mapper_pose_buf;
  std::queue<std_msgs::Header> mapper_header_buf;
  std::queue<std::pair<std_msgs::Header, torch::Tensor>> pose_msg_buf;
  std::thread mapper_thread, keyboard_thread;

  bool is_init_ = false;

  void register_subscriber(ros::NodeHandle &nh);
  void register_publisher(ros::NodeHandle &nh);

  template <typename DepthMsgT>
  void depth_callback(const DepthMsgT &_depth_msg);
  template <typename PoseMsgT> void pose_callback(const PoseMsgT &pose_msg);

  bool get_input(torch::Tensor &pose, std_msgs::Header &_header,
                 torch::Tensor &_xyz_rays_depths);
  void pub_pose(const torch::Tensor &_pose, const std_msgs::Header &_header);

  torch::Tensor train(const torch::Tensor &_pos,
                      const torch::Tensor &_xyz_rays_depths, int _opt_iter);

  void mapper_loop();
  void keyboard_loop();

  void visualization(const torch::Tensor &_pose, const torch::Tensor &_xyz,
                     const std_msgs::Header &_header);

  void export_checkpoint();
  void load_checkpoint(const std::string &_checkpoint_path);

  void save_mesh();
  void eval_mesh();

  void export_timing();

  bool end();
};