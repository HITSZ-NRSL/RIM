#pragma once

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>

#include <torch/torch.h>

#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <sensor_msgs/PointCloud2.h>
#include <std_msgs/Float32.h>
#include <visualization_msgs/MarkerArray.h>

#include <cuda_runtime_api.h>
#include <tinyply/tinyply.h>

#include "commons.h"

namespace utils {
// get cpu data memory usage
double get_cpu_mem_usage();

double get_gpu_mem_usage();

torch::Tensor downsample_point(const torch::Tensor &points, int ds_pt_num);

void extract_ray_depth_from_pcl(const torch::Tensor &pointcloud,
                                torch::Tensor &rays_d, torch::Tensor &depths,
                                float min_range = 0.01);

torch::Tensor pointcloud_to_tensor(PointCloudT &pointcloud);

void pointcloud_to_raydepth(PointCloudT &_pointcloud, int _ds_pt_num,
                            torch::Tensor &_rays_d, torch::Tensor &_depths,
                            float min_range = 0.01,
                            torch::Device device = torch::kCPU);

void sample_batch_pts(const torch::Tensor &_pts_rays_depths,
                      torch::Tensor &batch_pts_rays_depths,
                      int batch_pt_num = -1, int batch_type = 1, int iter = 0);

void sample_surface_pts(const torch::Tensor &_pts_rays_depths,
                        torch::Tensor &sample_pts, torch::Tensor &sample_dirs,
                        torch::Tensor &sample_gts, int surface_sample_num = 4,
                        float std = 0.1);

void sample_strat_pts(const torch::Tensor &_pts_rays_depths,
                      torch::Tensor &sample_pts, torch::Tensor &sample_dirs,
                      torch::Tensor &sample_gts, int strat_sample_num = 4,
                      float std = 0.1, float strat_near_ratio = 0.2,
                      float strat_far_ratio = 0.8);

torch::Tensor cal_nn_dist(const torch::Tensor &_input_pts,
                          const torch::Tensor &_target_pts);

void sample(torch::Tensor &_pts_rays_depths, RaySamples &samples,
            int surface_sample_num = 4, int free_sample_num = 4,
            float std = 0.1, float strat_near_ratio = 0.3,
            float strat_far_ratio = 0.7, int dist_type = 0);

torch::Tensor get_verteices(const torch::Tensor &xyz_index);

torch::Tensor get_width_verteices(const torch::Tensor &xyz_index,
                                  float width = 1.0);

torch::Tensor cal_tri_inter_coef(const torch::Tensor &xyz_weight);

torch::Tensor cal_inter_pair_coef(const torch::Tensor &vertex_sdf,
                                  const torch::Tensor &face_edge_pair_index,
                                  float iso_value = 0.0);

visualization_msgs::Marker get_vix_voxel_map(const torch::Tensor &_xyz,
                                             float voxel_size, float r = 1.0,
                                             float g = 1.0, float b = 1.0);

torch::Tensor quat_to_rot(const torch::Tensor &quat);

torch::Tensor rot_to_quat(const torch::Tensor &rotation);

torch::Tensor positional_encode(const torch::Tensor &xyz);

torch::Tensor meshgrid_2d(int width, int height,
                          torch::Device device = torch::kCPU);

torch::Tensor meshgrid_3d(float x_min, float x_max, float y_min, float y_max,
                          float z_min, float z_max, float resolution,
                          torch::Device &device);

torch::Tensor meshgrid_3d_slice(float x_min, float x_max, float y_min,
                                float y_max, float z_min, float z_max,
                                float resolution, torch::Device &device,
                                float slice_height);

torch::Tensor cal_face_normal(const torch::Tensor &face_xyz);

torch::Tensor cal_face_normal_color(const torch::Tensor &face_xyz);

visualization_msgs::Marker get_vis_shift_map(torch::Tensor _pos_W_M,
                                             float _x_min, float _x_max,
                                             float _y_min, float _y_max,
                                             float _z_min, float _z_max);

void export_to_ply(const torch::Tensor &_xyz, const std::string &output_path,
                   const std::string &_name = "all");
} // namespace utils