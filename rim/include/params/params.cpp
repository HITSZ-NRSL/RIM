#include "params.h"
#include "ros/package.h"
#include <opencv2/opencv.hpp>

bool k_shift_map;

bool k_online;
std::string k_calib_file, k_pose_file, k_depth_path, k_gt_structure_file;

std::string k_depth_topic, k_pose_topic;
int k_ds_pt_num, k_max_pt_num;
int k_bundle_frame_num;
int k_every_frame;

std::string k_output_path;
std::string k_package_path;

int k_depth_type;
int k_pose_msg_type; // 0:geometry_msgs::PoseStamped; 1:nav_msgs::Odometry

std::shared_ptr<torch::Device> p_device;
// parameter
float k_x_max, k_x_min, k_y_max, k_y_min, k_z_max, k_z_min, k_min_range;
float k_leaf_size;
int k_layer_num;
std::vector<float> voxel_sizes;
torch::Tensor t_voxel_sizes;
int k_iter_step, k_surface_sample_num, k_free_sample_num, k_out_sample_num,
    k_batch_pt_num, k_batch_type;
float k_sample_std, k_strat_near_ratio, k_strat_far_ratio;
bool k_use_outrange;
bool k_hist_pt;
bool k_outlier_remove;
double k_outlier_dist;

int k_decoder_freeze_frame_num;
int k_feat_dim;
int k_hidden_dim;
int k_input_dim;

// abalation parmaeter
bool mapper_init, mapper_update;
int k_supervise_mode; // 0:tsdf; 1:bce_sdf
float k_bce_sigma, k_truncated_dis;
int k_multifeat_type; // 0:concat; 1:sum
bool k_eikonal, k_positional_encoding;
float k_eikonal_weight;
int k_dist_type;

float k_lr;
bool k_zero_init;

float k_fx, k_fy, k_cx, k_cy;
torch::Tensor k_T_B_S;
int k_frame_rate;
float inv_frame_rate;

// visualization
bool k_large_scene;
int k_eval_mode;
bool k_save_normal;
bool k_skip_unconf;
int k_vis_frame_step;
int k_vis_batch_pt_num;
float k_vis_res, k_export_res, k_slice_height;
float k_downsample_mesh_size;

void print_files(const std::string &_file_path) {
  std::cout << "print_files: " << _file_path << '\n';
  std::ifstream file(_file_path);
  std::string str;
  while (std::getline(file, str)) {
    std::cout << str << '\n';
  }
  file.close();
  std::cout << "print_files end\n";
}

void read_params(const std::filesystem::path &_config_path,
                 const std::filesystem::path &_data_path) {
  // get now data and time
  std::time_t t = std::time(nullptr);
  std::tm tm = *std::localtime(&t);
  std::stringstream ss;
  ss << std::put_time(&tm, "%Y-%m-%d-%H-%M-%S");

  k_package_path = ros::package::getPath("neural_slam");
  k_output_path = k_package_path + "/output/" + ss.str();
  std::cout << "output_path: " << k_output_path << '\n';
  std::filesystem::create_directories(k_output_path);

  cv::FileStorage fsSettings(_config_path, cv::FileStorage::READ);
  if (!fsSettings.isOpened()) {
    std::cerr << "ERROR: Wrong path to settings: " << _config_path << "\n";
    exit(-1);
  }

  std::string params_file_path = k_output_path + "/parameters.txt";
  static std::ofstream param_file(params_file_path);

  param_file << "config_path: " << _config_path << '\n';
  param_file << "data_path: " << _data_path << '\n';

  param_file << "package_path: " << k_package_path << '\n';
  param_file << "output_path: " << k_output_path << '\n';

  /* Start reading parameters */

  fsSettings["shift_map"] >> k_shift_map;
  param_file << "shift_map: " << k_shift_map << "\n";

  fsSettings["online"] >> k_online;
  param_file << "k_online: " << k_online << "\n";
  if (k_online) {
    fsSettings["pose_msg_type"] >> k_pose_msg_type;
    param_file << "pose_msg_type: " << k_pose_msg_type << "\n";
    fsSettings["pose_topic"] >> k_pose_topic;
    param_file << "pose_topic: " << k_pose_topic << '\n';

    fsSettings["depth_type"] >> k_depth_type;
    param_file << "depth_type: " << k_depth_type << '\n';
    fsSettings["depth_topic"] >> k_depth_topic;
    param_file << "depth_topic: " << k_depth_topic << '\n';
    if (k_depth_type == 1) {
      fsSettings["fx"] >> k_fx;
      param_file << "fx: " << k_fx << '\n';
      fsSettings["fy"] >> k_fy;
      param_file << "fy: " << k_fy << '\n';
      fsSettings["cx"] >> k_cx;
      param_file << "cx: " << k_cx << '\n';
      fsSettings["cy"] >> k_cy;
      param_file << "cy: " << k_cy << '\n';
    }

    cv::Mat cv_T_B_S;
    fsSettings["T_B_S"] >> cv_T_B_S;
    cv_T_B_S.convertTo(cv_T_B_S, CV_32FC1);
    k_T_B_S = torch::from_blob(cv_T_B_S.data, {3, 4}, torch::kFloat32).clone();
    param_file << "T_B_S: \n" << k_T_B_S << '\n';
    bool inv_extrinsic;
    fsSettings["inv_extrinsic"] >> inv_extrinsic;
    param_file << "inv_extrinsic: " << inv_extrinsic << '\n';
    if (inv_extrinsic) {
      k_T_B_S = torch::cat({k_T_B_S, torch::tensor({{0., 0., 0., 1.}})}, 0);
      k_T_B_S = k_T_B_S.inverse().slice(0, 0, 3);
    }
  } else {
    if (!std::string(fsSettings["calib_file"]).empty()) {
      k_calib_file = _data_path / std::string(fsSettings["calib_file"]);
      param_file << "calib_file: " << k_calib_file << '\n';
    }
    k_pose_file = _data_path / std::string(fsSettings["pose_file"]);
    k_depth_path = _data_path / std::string(fsSettings["depth_path"]);
    param_file << "pose_file: " << k_pose_file << '\n';
    param_file << "depth_path: " << k_depth_path << '\n';
    k_gt_structure_file = std::string(fsSettings["gt_structure_file"]);
    param_file << "gt_structure_file: " << k_gt_structure_file << '\n';

    fsSettings["every_frame"] >> k_every_frame;
    param_file << "every_frame: " << k_every_frame << '\n';
  }
  fsSettings["frame_rate"] >> k_frame_rate;
  param_file << "frame_rate: " << k_frame_rate << '\n';
  inv_frame_rate = 1.0 / k_frame_rate;

  bool device_param;
  fsSettings["device_param"] >> device_param;
  param_file << "device_param: " << device_param << '\n';
  if (device_param) {
    p_device = std::make_shared<torch::Device>(
        torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);
  } else {
    p_device = std::make_shared<torch::Device>(torch::kCPU);
  }
  param_file << "Running Device: " << *p_device << '\n';

  fsSettings["x_max"] >> k_x_max;
  param_file << "x_max: " << k_x_max << '\n';
  fsSettings["x_min"] >> k_x_min;
  param_file << "x_min: " << k_x_min << '\n';
  fsSettings["y_max"] >> k_y_max;
  param_file << "y_max: " << k_y_max << '\n';
  fsSettings["y_min"] >> k_y_min;
  param_file << "y_min: " << k_y_min << '\n';
  fsSettings["z_max"] >> k_z_max;
  param_file << "z_max: " << k_z_max << '\n';
  fsSettings["z_min"] >> k_z_min;
  param_file << "z_min: " << k_z_min << '\n';
  fsSettings["min_range"] >> k_min_range;
  param_file << "min_range: " << k_min_range << '\n';

  fsSettings["leaf_sizes"] >> k_leaf_size;
  param_file << "leaf_sizes: " << k_leaf_size << '\n';
  fsSettings["layer_num"] >> k_layer_num;
  param_file << "layer_num: " << k_layer_num << '\n';

  t_voxel_sizes = torch::zeros({k_layer_num, 1, 1}, *p_device);
  voxel_sizes.resize(k_layer_num);
  for (int l = 0; l < k_layer_num; ++l) {
    voxel_sizes[l] = k_leaf_size * pow(2, k_layer_num - 1 - l);
    t_voxel_sizes[l][0][0] = voxel_sizes[l];
  }

  k_x_max =
      ceil((k_x_max - k_x_min) / voxel_sizes[0]) * voxel_sizes[0] + k_x_min;
  k_y_max =
      ceil((k_y_max - k_y_min) / voxel_sizes[0]) * voxel_sizes[0] + k_y_min;
  k_z_max =
      ceil((k_z_max - k_z_min) / voxel_sizes[0]) * voxel_sizes[0] + k_z_min;

  fsSettings["surface_sample_num"] >> k_surface_sample_num;
  param_file << "surface_sample_num: " << k_surface_sample_num << '\n';
  fsSettings["free_sample_num"] >> k_free_sample_num;
  param_file << "free_sample_num: " << k_free_sample_num << '\n';
  fsSettings["batch_pt_num"] >> k_batch_pt_num;
  param_file << "batch_pt_num: " << k_batch_pt_num << '\n';
  fsSettings["batch_type"] >> k_batch_type;
  param_file << "batch_type: " << k_batch_type << '\n';
  fsSettings["iter_step"] >> k_iter_step;
  param_file << "iter_step: " << k_iter_step << '\n';
  fsSettings["sample_std"] >> k_sample_std;
  param_file << "sample_std: " << k_sample_std << '\n';
  fsSettings["bce_sigma"] >> k_bce_sigma;
  param_file << "bce_sigma: " << k_bce_sigma << '\n';
  k_truncated_dis = 3 * k_bce_sigma;
  fsSettings["strat_near_ratio"] >> k_strat_near_ratio;
  param_file << "strat_near_ratio: " << k_strat_near_ratio << '\n';
  fsSettings["strat_far_ratio"] >> k_strat_far_ratio;
  param_file << "strat_far_ratio: " << k_strat_far_ratio << '\n';
  fsSettings["use_outrange"] >> k_use_outrange;
  param_file << "use_outrange: " << k_use_outrange << '\n';
  fsSettings["out_sample_num"] >> k_out_sample_num;
  param_file << "out_sample_num: " << k_out_sample_num << '\n';
  fsSettings["hist_pt"] >> k_hist_pt;
  param_file << "hist_pt: " << k_hist_pt << '\n';
  fsSettings["outlier_remove"] >> k_outlier_remove;
  param_file << "outlier_remove: " << k_outlier_remove << '\n';
  fsSettings["outlier_dist"] >> k_outlier_dist;
  param_file << "outlier_dist: " << k_outlier_dist << '\n';

  fsSettings["multifeat_type"] >> k_multifeat_type;
  param_file << "multifeat_type: " << k_multifeat_type << '\n';
  fsSettings["positional_encoding"] >> k_positional_encoding;
  param_file << "positional_encoding: " << k_positional_encoding << '\n';

  fsSettings["decoder_freeze_frame_num"] >> k_decoder_freeze_frame_num;
  param_file << "decoder_freeze_frame_num: " << k_decoder_freeze_frame_num
             << '\n';
  fsSettings["feat_dim"] >> k_feat_dim;
  param_file << "feat_dim: " << k_feat_dim << '\n';
  fsSettings["hidden_dim"] >> k_hidden_dim;
  param_file << "hidden_dim: " << k_hidden_dim << '\n';

  if (k_multifeat_type == 0) {
    k_input_dim = k_feat_dim * k_layer_num;
  } else {
    k_input_dim = k_feat_dim;
  }
  if (k_positional_encoding)
    k_input_dim += 33;

  fsSettings["lr"] >> k_lr;
  param_file << "lr: " << k_lr << '\n';
  fsSettings["zero_init"] >> k_zero_init;
  param_file << "zero_init: " << k_zero_init << '\n';

  fsSettings["ds_pt_num"] >> k_ds_pt_num;
  param_file << "ds_pt_num: " << k_ds_pt_num << '\n';
  fsSettings["max_pt_num"] >> k_max_pt_num;
  param_file << "max_pt_num: " << k_max_pt_num << '\n';

  fsSettings["supervise_mode"] >> k_supervise_mode;
  param_file << "supervise_mode: " << k_supervise_mode << '\n';
  fsSettings["eikonal"] >> k_eikonal;
  param_file << "eikonal: " << k_eikonal << '\n';
  if (k_eikonal) {
    fsSettings["eikonal_weight"] >> k_eikonal_weight;
    param_file << "eikonal_weight: " << k_eikonal_weight << '\n';
  }

  fsSettings["dist_type"] >> k_dist_type;
  param_file << "dist_type: " << k_dist_type << '\n';
  fsSettings["large_scene"] >> k_large_scene;
  param_file << "large_scene: " << k_large_scene << '\n';
  fsSettings["eval_mode"] >> k_eval_mode;
  param_file << "eval_mode: " << k_eval_mode << '\n';
  fsSettings["save_normal"] >> k_save_normal;
  param_file << "save_normal: " << k_save_normal << '\n';

  fsSettings["vis_frame_step"] >> k_vis_frame_step;
  param_file << "vis_frame_step: " << k_vis_frame_step << '\n';
  fsSettings["vis_batch_pt_num"] >> k_vis_batch_pt_num;
  param_file << "vis_batch_pt_num: " << k_vis_batch_pt_num << '\n';
  fsSettings["vis_resolution"] >> k_vis_res;
  param_file << "vis_resolution: " << k_vis_res << '\n';

  fsSettings["export_resolution"] >> k_export_res;
  param_file << "export_resolution: " << k_export_res << '\n';
  fsSettings["downsample_mesh_size"] >> k_downsample_mesh_size;
  param_file << "downsample_mesh_size: " << k_downsample_mesh_size << '\n';

  fsSettings["skip_unconf"] >> k_skip_unconf;
  param_file << "skip_unconf: " << k_skip_unconf << '\n';

  fsSettings["slice_height"] >> k_slice_height;
  param_file << "slice_height: " << k_slice_height << '\n';
  fsSettings.release();
  param_file.close();
  print_files(params_file_path);
}