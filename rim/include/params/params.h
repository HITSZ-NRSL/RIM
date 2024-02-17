#pragma once

#include <filesystem>
#include <torch/torch.h>

extern bool k_shift_map;

extern bool k_online;
extern std::string k_calib_file, k_pose_file, k_depth_path, k_gt_structure_file;

extern std::string k_depth_topic, k_pose_topic;
extern int k_ds_pt_num, k_max_pt_num;
extern int k_every_frame;

extern std::string k_output_path;

extern int k_depth_type;
extern int
    k_pose_msg_type; // 0:geometry_msgs::PoseStamped; 1:nav_msgs::Odometry

extern std::shared_ptr<torch::Device> p_device;
// parameter
extern float k_x_max, k_x_min, k_y_max, k_y_min, k_z_max, k_z_min, k_min_range;
extern float k_leaf_size;
extern int k_layer_num;
extern std::vector<float> voxel_sizes;
extern torch::Tensor t_voxel_sizes;
extern int k_iter_step, k_surface_sample_num, k_free_sample_num,
    k_out_sample_num, k_batch_pt_num, k_batch_type;
extern float k_sample_std, k_strat_near_ratio, k_strat_far_ratio;
extern bool k_use_outrange;
extern bool k_hist_pt;
extern bool k_outlier_remove;
extern double k_outlier_dist;

extern int k_decoder_freeze_frame_num;
extern int k_feat_dim;
extern int k_hidden_dim;
extern int k_input_dim;

// abalation parmaeter
extern bool mapper_init, mapper_update;
extern int k_supervise_mode; // 0:tsdf; 1:bce_sdf
extern float k_bce_sigma, k_truncated_dis;
extern int k_multifeat_type; // 0:concat; 1:sum
extern bool k_eikonal, k_positional_encoding;
extern float k_eikonal_weight;
extern int k_dist_type;

extern float k_lr;
extern bool k_zero_init;

extern float k_fx, k_fy, k_cx, k_cy;
extern torch::Tensor k_T_B_S;
extern int k_frame_rate;
extern float inv_frame_rate;

// visualization
extern bool k_large_scene;
extern int k_eval_mode;
extern bool k_save_normal;
extern bool k_skip_unconf;
extern int k_vis_frame_step;
extern int k_vis_batch_pt_num;
extern float k_vis_res, k_export_res, k_slice_height;
extern float k_downsample_mesh_size;

void read_params(const std::filesystem::path &_config_path,
                 const std::filesystem::path &_data_path);