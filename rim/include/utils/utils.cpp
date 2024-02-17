#include "utils.h"

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

using namespace std;

namespace utils {

// get cpu data memory usage
double get_cpu_mem_usage() {
  FILE *file = fopen("/proc/self/status", "r");
  int result = -1;
  char line[128];

  while (fgets(line, 128, file) != nullptr) {
    if (strncmp(line, "VmRSS:", 6) == 0) {
      int len = strlen(line);

      const char *p = line;
      for (; std::isdigit(*p) == false; ++p) {
      }

      line[len - 3] = 0;
      result = atoi(p);

      break;
    }
  }
  fclose(file);

  std::cout << "Now used CPU memory " << result / 1024.0 / 1024.0 << "  GB\n";

  return (result / 1024.0 / 1024.0);
}

double get_gpu_mem_usage() {
  size_t free_byte;
  size_t total_byte;

  cudaError_t cuda_status = cudaMemGetInfo(&free_byte, &total_byte);

  if (cudaSuccess != cuda_status) {
    printf("Error: cudaMemGetInfo fails, %s \n",
           cudaGetErrorString(cuda_status));
    exit(1);
  }

  auto free_db = (double)free_byte;
  auto total_db = (double)total_byte;
  double used_db = (total_db - free_db) / 1024.0 / 1024.0 / 1024.0;
  std::cout << "Now used GPU memory " << used_db << "  GB\n";
  return used_db;
}

torch::Tensor downsample_point(const torch::Tensor &points, int ds_pt_num) {
  if (points.size(0) > ds_pt_num) {
    int step = points.size(0) / ds_pt_num + 1.0;
    return points.slice(0, 0, -1, step);
  } else {
    return points;
  }
}

void extract_ray_depth_from_pcl(const torch::Tensor &pointcloud,
                                torch::Tensor &rays_d, torch::Tensor &depths,
                                float min_range) {
  depths = pointcloud.norm(2, 1, true);
  rays_d = pointcloud.div(depths);

  // filter out nan
  auto min_mask = (depths.squeeze() > min_range).nonzero().squeeze();
  depths = depths.index({min_mask});
  rays_d = rays_d.index({min_mask});
}

/**
 * @description: PointXYZ: 4; PointXYZRGB(A): 8
 * @param {PointCloudT} &pointcloud
 * @return {*}
 */
torch::Tensor pointcloud_to_tensor(PointCloudT &pointcloud) {
  return torch::from_blob(pointcloud.points.data(),
                          {(long)pointcloud.points.size(), 4}, torch::kFloat)
      .slice(1, 0, 3);
}

void pointcloud_to_raydepth(PointCloudT &_pointcloud, int _ds_pt_num,
                            torch::Tensor &_rays_d, torch::Tensor &_depths,
                            float min_range, torch::Device device) {

  // [n,3]
  auto xyz = pointcloud_to_tensor(_pointcloud).to(device);

  // [n,3],[n,3],[n,1]
  extract_ray_depth_from_pcl(xyz, _rays_d, _depths, min_range);
  _rays_d = downsample_point(_rays_d, _ds_pt_num);
  _depths = downsample_point(_depths, _ds_pt_num);
}

void sample_batch_pts(const torch::Tensor &_pts_rays_depths,
                      torch::Tensor &batch_pts_rays_depths, int batch_pt_num,
                      int batch_type, int iter) {
  if (batch_pt_num > 0 && _pts_rays_depths.size(0) > batch_pt_num) {
    if (batch_type == 0) {
      auto sample_idx = torch::randint(0, _pts_rays_depths.size(0),
                                       {batch_pt_num}, torch::kLong)
                            .to(_pts_rays_depths.device());
      batch_pts_rays_depths = _pts_rays_depths.index({sample_idx});
    } else if (batch_type == 1) {
      int start = (batch_pt_num * iter) % _pts_rays_depths.size(0);
      int end = start + batch_pt_num;
      batch_pts_rays_depths = _pts_rays_depths.slice(0, start, end);
      if (end > _pts_rays_depths.size(0)) {
        auto tmp_end = end - _pts_rays_depths.size(0);
        batch_pts_rays_depths = torch::cat(
            {batch_pts_rays_depths, _pts_rays_depths.slice(0, 0, tmp_end)}, 0);
      }
    } else if (batch_type == 2) {
      int step = _pts_rays_depths.size(0) / batch_pt_num + 1.0;
      int rand = torch::randint(0, step, {1}, torch::kLong).item<int>();
      batch_pts_rays_depths = _pts_rays_depths.slice(0, rand, -1, step);
    }
  } else {
    batch_pts_rays_depths = _pts_rays_depths;
  }
}

void sample_surface_pts(const torch::Tensor &_pts_rays_depths,
                        torch::Tensor &sample_pts, torch::Tensor &sample_dirs,
                        torch::Tensor &sample_gts, int surface_sample_num,
                        float std) {
  // sample points along rays_d within voxel
  /// [n,k,1]
  sample_gts = torch::randn({_pts_rays_depths.size(0), surface_sample_num, 1},
                            _pts_rays_depths.device()) *
               std;
  /// [n,k,3]->[n*k,3]
  sample_pts = (_pts_rays_depths.slice(1, 0, 3).unsqueeze(1) -
                _pts_rays_depths.slice(1, 3, 6).unsqueeze(1) * sample_gts);
  sample_dirs = _pts_rays_depths.slice(1, 3, 6).unsqueeze(1).repeat(
      {1, surface_sample_num, 1});
}

void sample_strat_pts(const torch::Tensor &_pts_rays_depths,
                      torch::Tensor &sample_pts, torch::Tensor &sample_dirs,
                      torch::Tensor &sample_gts, int strat_sample_num,
                      float std, float strat_near_ratio,
                      float strat_far_ratio) {

  auto device = _pts_rays_depths.device();
  // sample points along rays_d within voxel
  /// [n,k,1]
  auto linspace = torch::linspace(1 - strat_far_ratio, 1 - strat_near_ratio,
                                  strat_sample_num, device)
                      .view({1, strat_sample_num});
  sample_gts = linspace.expand({_pts_rays_depths.size(0), strat_sample_num});
  auto strat_sample_rand = std * torch::randn({1, strat_sample_num, 1}, device);
  sample_gts = (_pts_rays_depths.slice(1, 6, 7) * sample_gts)
                   .view({-1, strat_sample_num, 1}) +
               strat_sample_rand;
  // [n,k,3]->[n*k,3]
  sample_pts = (_pts_rays_depths.slice(1, 0, 3).view({-1, 1, 3}) -
                _pts_rays_depths.slice(1, 3, 6).unsqueeze(1) * sample_gts);
  sample_dirs = _pts_rays_depths.slice(1, 3, 6).unsqueeze(1).repeat(
      {1, strat_sample_num, 1});
}

torch::Tensor cal_nn_dist(const torch::Tensor &_input_pts,
                          const torch::Tensor &_target_pts) {
  auto pts_dim = _target_pts.size(1);
  auto dist =
      (_target_pts.view({1, -1, pts_dim}) - _input_pts.view({-1, 1, pts_dim}))
          .norm(2, 2);
  return get<0>(dist.min(1));
}

void sample(torch::Tensor &_pts_rays_depths, RaySamples &samples,
            int surface_sample_num, int free_sample_num, float std,
            float strat_near_ratio, float strat_far_ratio, int dist_type) {
  if (surface_sample_num > 0) {
    sample_surface_pts(_pts_rays_depths, samples.xyz, samples.direction,
                       samples.ray_sdf, surface_sample_num, std);
  }

  if (free_sample_num > 0) {
    torch::Tensor void_pts, void_dirs, void_gts;
    sample_strat_pts(_pts_rays_depths, void_pts, void_dirs, void_gts,
                     free_sample_num, std, strat_near_ratio, strat_far_ratio);

    samples.xyz = torch::cat({samples.xyz, void_pts}, 1);
    samples.ray_sdf = torch::cat({samples.ray_sdf, void_gts}, 1);
    samples.direction = torch::cat({samples.direction, void_dirs}, 1);
  }

  if (surface_sample_num < 1 && free_sample_num < 1) {
    samples.xyz = _pts_rays_depths.slice(1, 0, 3);
    samples.ray_sdf = torch::zeros_like(_pts_rays_depths.slice(1, 6, 7));
    samples.direction = _pts_rays_depths.slice(1, 3, 6);
  }

  if (dist_type) {
    // nearest input point distance
    // auto diff = _pts.view({1, -1, 3}) - sample_pts.view({-1, 1, 3});
    // auto dist = diff.norm(2, 2);
    // dist = get<0>(dist.min(1));
    auto dist = cal_nn_dist(samples.xyz, _pts_rays_depths.slice(1, 0, 3));
    samples.ray_sdf = dist * samples.ray_sdf.view({-1}).sign();
  }

  samples.xyz = torch::cat(
      {samples.xyz.view({-1, 3}), _pts_rays_depths.slice(1, 0, 3)}, 0);
  samples.ray_sdf =
      torch::cat({samples.ray_sdf.view({-1, 1}),
                  torch::zeros_like(_pts_rays_depths.slice(1, 6, 7))},
                 0);
  samples.direction = torch::cat(
      {samples.direction.view({-1, 3}), _pts_rays_depths.slice(1, 3, 6)}, 0);
}

torch::Tensor get_verteices(const torch::Tensor &xyz_index) {
  /**
   * @description:
   * @return [n,8,3]
   */
  auto device = xyz_index.device();
  auto tensor_option =
      torch::TensorOptions().dtype(torch::kLong).device(device);
  return torch::stack({xyz_index,
                       xyz_index + torch::tensor({{0, 0, 1}}, tensor_option),
                       xyz_index + torch::tensor({{0, 1, 0}}, tensor_option),
                       xyz_index + torch::tensor({{0, 1, 1}}, tensor_option),
                       xyz_index + torch::tensor({{1, 0, 0}}, tensor_option),
                       xyz_index + torch::tensor({{1, 0, 1}}, tensor_option),
                       xyz_index + torch::tensor({{1, 1, 0}}, tensor_option),
                       xyz_index + torch::tensor({{1, 1, 1}}, tensor_option)},
                      1);
}

torch::Tensor get_width_verteices(const torch::Tensor &xyz_index, float width) {
  auto device = xyz_index.device();
  auto xyz_001 = xyz_index + width * torch::tensor({{0., 0., 1.}}, device); // 1
  auto xyz_010 = xyz_index + width * torch::tensor({{0., 1., 0.}}, device); // 2
  auto xyz_011 = xyz_index + width * torch::tensor({{0., 1., 1.}}, device); // 3
  auto xyz_100 = xyz_index + width * torch::tensor({{1., 0., 0.}}, device); // 4
  auto xyz_101 = xyz_index + width * torch::tensor({{1., 0., 1.}}, device); // 5
  auto xyz_110 = xyz_index + width * torch::tensor({{1., 1., 0.}}, device); // 6
  auto xyz_111 = xyz_index + width * torch::tensor({{1., 1., 1.}}, device); // 7
  return torch::stack({xyz_index, xyz_001, xyz_010, xyz_011, xyz_100, xyz_101,
                       xyz_110, xyz_111},
                      1);
}

torch::Tensor cal_tri_inter_coef(const torch::Tensor &xyz_weight) {
  auto xyz_weight_x = xyz_weight.select(1, 0);
  auto xyz_weight_y = xyz_weight.select(1, 1);
  auto xyz_weight_z = xyz_weight.select(1, 2);
  auto coef_000 = (1 - xyz_weight_x) * (1 - xyz_weight_y) * (1 - xyz_weight_z);
  auto coef_001 = (1 - xyz_weight_x) * (1 - xyz_weight_y) * xyz_weight_z;
  auto coef_010 = (1 - xyz_weight_x) * xyz_weight_y * (1 - xyz_weight_z);
  auto coef_011 = (1 - xyz_weight_x) * xyz_weight_y * xyz_weight_z;
  auto coef_100 = xyz_weight_x * (1 - xyz_weight_y) * (1 - xyz_weight_z);
  auto coef_101 = xyz_weight_x * (1 - xyz_weight_y) * xyz_weight_z;
  auto coef_110 = xyz_weight_x * xyz_weight_y * (1 - xyz_weight_z);
  auto coef_111 = xyz_weight_x * xyz_weight_y * xyz_weight_z;
  return torch::stack({coef_000, coef_001, coef_010, coef_011, coef_100,
                       coef_101, coef_110, coef_111},
                      1);
}

torch::Tensor cal_inter_pair_coef(const torch::Tensor &vertex_sdf,
                                  const torch::Tensor &face_edge_pair_index,
                                  float iso_value) {
  auto face_edge_pair_sdf =
      vertex_sdf.index({face_edge_pair_index.view({-1})}).view({-1, 2});
  auto face_edge_coef =
      (iso_value - face_edge_pair_sdf.select(1, 0)) /
      (face_edge_pair_sdf.select(1, 1) - face_edge_pair_sdf.select(1, 0));
  return face_edge_coef.view({-1, 3});
}

visualization_msgs::Marker get_vix_voxel_map(const torch::Tensor &_xyz,
                                             float voxel_size, float r, float g,
                                             float b) {
  auto xyz_idx_uniq = (_xyz.squeeze() / voxel_size).floor().to(torch::kLong);
  xyz_idx_uniq = std::get<0>(torch::unique_dim(xyz_idx_uniq, 0));
  xyz_idx_uniq = (xyz_idx_uniq * voxel_size);
  /// [n, 8, 3]
  auto voxel_vertex_xyz =
      get_width_verteices(xyz_idx_uniq, voxel_size).to(torch::kCPU);
  auto voxel_center_xyz_a = voxel_vertex_xyz.accessor<float, 3>();

  visualization_msgs::Marker marker;
  marker.points.resize(voxel_vertex_xyz.size(0) * 24);
  marker.type = visualization_msgs::Marker::LINE_LIST;
  marker.action = visualization_msgs::Marker::ADD;
  marker.color.a = 1.0;
  marker.color.r = r;
  marker.color.g = g;
  marker.color.b = b;
  marker.scale.x = 0.01;
#pragma omp parallel for
  for (int i = 0; i < voxel_center_xyz_a.size(0); i++) {
    marker.points[i * 24 + 0].x = voxel_center_xyz_a[i][0][0];
    marker.points[i * 24 + 0].y = voxel_center_xyz_a[i][0][1];
    marker.points[i * 24 + 0].z = voxel_center_xyz_a[i][0][2];
    marker.points[i * 24 + 1].x = voxel_center_xyz_a[i][1][0];
    marker.points[i * 24 + 1].y = voxel_center_xyz_a[i][1][1];
    marker.points[i * 24 + 1].z = voxel_center_xyz_a[i][1][2];

    marker.points[i * 24 + 2].x = voxel_center_xyz_a[i][1][0];
    marker.points[i * 24 + 2].y = voxel_center_xyz_a[i][1][1];
    marker.points[i * 24 + 2].z = voxel_center_xyz_a[i][1][2];
    marker.points[i * 24 + 3].x = voxel_center_xyz_a[i][3][0];
    marker.points[i * 24 + 3].y = voxel_center_xyz_a[i][3][1];
    marker.points[i * 24 + 3].z = voxel_center_xyz_a[i][3][2];

    marker.points[i * 24 + 4].x = voxel_center_xyz_a[i][3][0];
    marker.points[i * 24 + 4].y = voxel_center_xyz_a[i][3][1];
    marker.points[i * 24 + 4].z = voxel_center_xyz_a[i][3][2];
    marker.points[i * 24 + 5].x = voxel_center_xyz_a[i][2][0];
    marker.points[i * 24 + 5].y = voxel_center_xyz_a[i][2][1];
    marker.points[i * 24 + 5].z = voxel_center_xyz_a[i][2][2];

    marker.points[i * 24 + 6].x = voxel_center_xyz_a[i][2][0];
    marker.points[i * 24 + 6].y = voxel_center_xyz_a[i][2][1];
    marker.points[i * 24 + 6].z = voxel_center_xyz_a[i][2][2];
    marker.points[i * 24 + 7].x = voxel_center_xyz_a[i][0][0];
    marker.points[i * 24 + 7].y = voxel_center_xyz_a[i][0][1];
    marker.points[i * 24 + 7].z = voxel_center_xyz_a[i][0][2];

    marker.points[i * 24 + 8].x = voxel_center_xyz_a[i][4][0];
    marker.points[i * 24 + 8].y = voxel_center_xyz_a[i][4][1];
    marker.points[i * 24 + 8].z = voxel_center_xyz_a[i][4][2];
    marker.points[i * 24 + 9].x = voxel_center_xyz_a[i][5][0];
    marker.points[i * 24 + 9].y = voxel_center_xyz_a[i][5][1];
    marker.points[i * 24 + 9].z = voxel_center_xyz_a[i][5][2];

    marker.points[i * 24 + 10].x = voxel_center_xyz_a[i][5][0];
    marker.points[i * 24 + 10].y = voxel_center_xyz_a[i][5][1];
    marker.points[i * 24 + 10].z = voxel_center_xyz_a[i][5][2];
    marker.points[i * 24 + 11].x = voxel_center_xyz_a[i][7][0];
    marker.points[i * 24 + 11].y = voxel_center_xyz_a[i][7][1];
    marker.points[i * 24 + 11].z = voxel_center_xyz_a[i][7][2];

    marker.points[i * 24 + 12].x = voxel_center_xyz_a[i][7][0];
    marker.points[i * 24 + 12].y = voxel_center_xyz_a[i][7][1];
    marker.points[i * 24 + 12].z = voxel_center_xyz_a[i][7][2];
    marker.points[i * 24 + 13].x = voxel_center_xyz_a[i][6][0];
    marker.points[i * 24 + 13].y = voxel_center_xyz_a[i][6][1];
    marker.points[i * 24 + 13].z = voxel_center_xyz_a[i][6][2];

    marker.points[i * 24 + 14].x = voxel_center_xyz_a[i][6][0];
    marker.points[i * 24 + 14].y = voxel_center_xyz_a[i][6][1];
    marker.points[i * 24 + 14].z = voxel_center_xyz_a[i][6][2];
    marker.points[i * 24 + 15].x = voxel_center_xyz_a[i][4][0];
    marker.points[i * 24 + 15].y = voxel_center_xyz_a[i][4][1];
    marker.points[i * 24 + 15].z = voxel_center_xyz_a[i][4][2];

    marker.points[i * 24 + 16].x = voxel_center_xyz_a[i][0][0];
    marker.points[i * 24 + 16].y = voxel_center_xyz_a[i][0][1];
    marker.points[i * 24 + 16].z = voxel_center_xyz_a[i][0][2];
    marker.points[i * 24 + 17].x = voxel_center_xyz_a[i][4][0];
    marker.points[i * 24 + 17].y = voxel_center_xyz_a[i][4][1];
    marker.points[i * 24 + 17].z = voxel_center_xyz_a[i][4][2];

    marker.points[i * 24 + 18].x = voxel_center_xyz_a[i][1][0];
    marker.points[i * 24 + 18].y = voxel_center_xyz_a[i][1][1];
    marker.points[i * 24 + 18].z = voxel_center_xyz_a[i][1][2];
    marker.points[i * 24 + 19].x = voxel_center_xyz_a[i][5][0];
    marker.points[i * 24 + 19].y = voxel_center_xyz_a[i][5][1];
    marker.points[i * 24 + 19].z = voxel_center_xyz_a[i][5][2];

    marker.points[i * 24 + 20].x = voxel_center_xyz_a[i][2][0];
    marker.points[i * 24 + 20].y = voxel_center_xyz_a[i][2][1];
    marker.points[i * 24 + 20].z = voxel_center_xyz_a[i][2][2];
    marker.points[i * 24 + 21].x = voxel_center_xyz_a[i][6][0];
    marker.points[i * 24 + 21].y = voxel_center_xyz_a[i][6][1];
    marker.points[i * 24 + 21].z = voxel_center_xyz_a[i][6][2];

    marker.points[i * 24 + 22].x = voxel_center_xyz_a[i][3][0];
    marker.points[i * 24 + 22].y = voxel_center_xyz_a[i][3][1];
    marker.points[i * 24 + 22].z = voxel_center_xyz_a[i][3][2];
    marker.points[i * 24 + 23].x = voxel_center_xyz_a[i][7][0];
    marker.points[i * 24 + 23].y = voxel_center_xyz_a[i][7][1];
    marker.points[i * 24 + 23].z = voxel_center_xyz_a[i][7][2];
  }
  return marker;
}

torch::Tensor quat_to_rot(const torch::Tensor &quat) {
  auto two_s = 2.0 / (quat * quat).sum(-1);
  auto rot = torch::zeros({3, 3}, quat.dtype()).to(quat.device());
  rot[0][0] = 1 - two_s * (quat[2] * quat[2] + quat[3] * quat[3]);
  rot[0][1] = two_s * (quat[1] * quat[2] - quat[3] * quat[0]);
  rot[0][2] = two_s * (quat[1] * quat[3] + quat[2] * quat[0]);
  rot[1][0] = two_s * (quat[1] * quat[2] + quat[3] * quat[0]);
  rot[1][1] = 1 - two_s * (quat[1] * quat[1] + quat[3] * quat[3]);
  rot[1][2] = two_s * (quat[2] * quat[3] - quat[1] * quat[0]);
  rot[2][0] = two_s * (quat[1] * quat[3] - quat[2] * quat[0]);
  rot[2][1] = two_s * (quat[2] * quat[3] + quat[1] * quat[0]);
  rot[2][2] = 1 - two_s * (quat[1] * quat[1] + quat[2] * quat[2]);
  return rot;
}

torch::Tensor rot_to_quat(const torch::Tensor &rotation) {
  // w,x,y,z
  auto quat = torch::zeros(4, rotation.dtype()).to(rotation.device());
  quat[0] =
      0.5 * torch::sqrt(1 + rotation[0][0] + rotation[1][1] + rotation[2][2]);
  quat[1] = 0.25 * (rotation[2][1] - rotation[1][2]) / quat[0];
  quat[2] = 0.25 * (rotation[0][2] - rotation[2][0]) / quat[0];
  quat[3] = 0.25 * (rotation[1][0] - rotation[0][1]) / quat[0];
  return quat;
}

torch::Tensor positional_encode(const torch::Tensor &xyz) {
  // torch::Tensor xyz_pos_enc = xyz;
  // torch::Tensor xyz_expo = xyz.clone();
  // for (int i = 0; i <= 4; i++) {
  //   xyz_pos_enc = torch::cat(
  //       {xyz_pos_enc, torch::sin(xyz_expo), torch::cos(xyz_expo)}, -1);
  //   xyz_expo = 2 * xyz_expo;
  // }
  // return xyz_pos_enc;
  return torch::cat({xyz, torch::sin(xyz), torch::cos(xyz), torch::sin(2 * xyz),
                     torch::cos(2 * xyz), torch::sin(4 * xyz),
                     torch::cos(4 * xyz), torch::sin(8 * xyz),
                     torch::cos(8 * xyz), torch::sin(16 * xyz),
                     torch::cos(16 * xyz)},
                    -1);
}

torch::Tensor meshgrid_2d(int width, int height, torch::Device device) {
  auto u = torch::arange(0, width, device);
  auto v = torch::arange(0, height, device);
  auto vu_meshgrid = torch::meshgrid({v, u});
  return torch::stack({vu_meshgrid[0], vu_meshgrid[1]}, -1).view({-1, 2});
}

torch::Tensor meshgrid_3d(float x_min, float x_max, float y_min, float y_max,
                          float z_min, float z_max, float resolution,
                          torch::Device &device) {
  auto x = torch::arange(x_min, x_max - 1e-6, resolution, device);
  auto y = torch::arange(y_min, y_max - 1e-6, resolution, device);

  torch::Tensor z;
  if (z_max <= z_min) {
    z = torch::tensor(z_min, device);
  } else {
    z = torch::arange(z_min, z_max - 1e-6, resolution, device);
  }
  auto xyz_seperate = torch::meshgrid({x, y, z});
  return torch::cat({xyz_seperate[0].unsqueeze(-1),
                     xyz_seperate[1].unsqueeze(-1),
                     xyz_seperate[2].unsqueeze(-1)},
                    -1);
}

torch::Tensor meshgrid_3d_slice(float x_min, float x_max, float y_min,
                                float y_max, float z_min, float z_max,
                                float resolution, torch::Device &device,
                                float slice_height) {
  auto x = torch::arange(x_min, x_max - 1e-6, resolution, device);
  auto y = torch::arange(y_min, y_max - 1e-6, resolution, device);

  torch::Tensor z = torch::tensor(slice_height, device);
  auto xyz_seperate = torch::meshgrid({x, y, z});
  return torch::cat({xyz_seperate[0].unsqueeze(-1),
                     xyz_seperate[1].unsqueeze(-1),
                     xyz_seperate[2].unsqueeze(-1)},
                    -1);
}

torch::Tensor cal_face_normal(const torch::Tensor &face_xyz) {
  // cal normal
  /// [n,3]
  auto v1 = face_xyz.select(1, 1) - face_xyz.select(1, 0);
  auto v2 = face_xyz.select(1, 2) - face_xyz.select(1, 1);
  auto face_normal = v1.cross(v2, 1);
  return face_normal / face_normal.norm(2, 1, true);
}

torch::Tensor cal_face_normal_color(const torch::Tensor &face_xyz) {
  // https://zhuanlan.zhihu.com/p/575404558
  return cal_face_normal(face_xyz) / 2.0 + 0.5;
}

visualization_msgs::Marker get_vis_shift_map(torch::Tensor _pos_W_M,
                                             float _x_min, float _x_max,
                                             float _y_min, float _y_max,
                                             float _z_min, float _z_max) {
  auto t_W_M_cpu = _pos_W_M.cpu();
  auto t_W_M_a = t_W_M_cpu.accessor<float, 2>();

  visualization_msgs::Marker marker;
  marker.points.resize(24);
  marker.type = visualization_msgs::Marker::LINE_LIST;
  marker.action = visualization_msgs::Marker::ADD;
  marker.color.a = 1.0;
  marker.color.r = 1.0;
  marker.color.g = 1.0;
  marker.color.b = 1.0;
  marker.scale.x = 0.01;
  {
    marker.points[0].x = t_W_M_a[0][0] + _x_min;
    marker.points[0].y = t_W_M_a[0][1] + _y_min;
    marker.points[0].z = t_W_M_a[0][2] + _z_min;
    marker.points[1].x = t_W_M_a[0][0] + _x_max;
    marker.points[1].y = t_W_M_a[0][1] + _y_min;
    marker.points[1].z = t_W_M_a[0][2] + _z_min;

    marker.points[2].x = t_W_M_a[0][0] + _x_max;
    marker.points[2].y = t_W_M_a[0][1] + _y_min;
    marker.points[2].z = t_W_M_a[0][2] + _z_min;
    marker.points[3].x = t_W_M_a[0][0] + _x_max;
    marker.points[3].y = t_W_M_a[0][1] + _y_max;
    marker.points[3].z = t_W_M_a[0][2] + _z_min;

    marker.points[4].x = t_W_M_a[0][0] + _x_max;
    marker.points[4].y = t_W_M_a[0][1] + _y_max;
    marker.points[4].z = t_W_M_a[0][2] + _z_min;
    marker.points[5].x = t_W_M_a[0][0] + _x_min;
    marker.points[5].y = t_W_M_a[0][1] + _y_max;
    marker.points[5].z = t_W_M_a[0][2] + _z_min;

    marker.points[6].x = t_W_M_a[0][0] + _x_min;
    marker.points[6].y = t_W_M_a[0][1] + _y_max;
    marker.points[6].z = t_W_M_a[0][2] + _z_min;
    marker.points[7].x = t_W_M_a[0][0] + _x_min;
    marker.points[7].y = t_W_M_a[0][1] + _y_min;
    marker.points[7].z = t_W_M_a[0][2] + _z_min;

    marker.points[8].x = t_W_M_a[0][0] + _x_min;
    marker.points[8].y = t_W_M_a[0][1] + _y_min;
    marker.points[8].z = t_W_M_a[0][2] + _z_max;
    marker.points[9].x = t_W_M_a[0][0] + _x_max;
    marker.points[9].y = t_W_M_a[0][1] + _y_min;
    marker.points[9].z = t_W_M_a[0][2] + _z_max;

    marker.points[10].x = t_W_M_a[0][0] + _x_max;
    marker.points[10].y = t_W_M_a[0][1] + _y_min;
    marker.points[10].z = t_W_M_a[0][2] + _z_max;
    marker.points[11].x = t_W_M_a[0][0] + _x_max;
    marker.points[11].y = t_W_M_a[0][1] + _y_max;
    marker.points[11].z = t_W_M_a[0][2] + _z_max;

    marker.points[12].x = t_W_M_a[0][0] + _x_max;
    marker.points[12].y = t_W_M_a[0][1] + _y_max;
    marker.points[12].z = t_W_M_a[0][2] + _z_max;
    marker.points[13].x = t_W_M_a[0][0] + _x_min;
    marker.points[13].y = t_W_M_a[0][1] + _y_max;
    marker.points[13].z = t_W_M_a[0][2] + _z_max;

    marker.points[14].x = t_W_M_a[0][0] + _x_min;
    marker.points[14].y = t_W_M_a[0][1] + _y_max;
    marker.points[14].z = t_W_M_a[0][2] + _z_max;
    marker.points[15].x = t_W_M_a[0][0] + _x_min;
    marker.points[15].y = t_W_M_a[0][1] + _y_min;
    marker.points[15].z = t_W_M_a[0][2] + _z_max;

    marker.points[16].x = t_W_M_a[0][0] + _x_min;
    marker.points[16].y = t_W_M_a[0][1] + _y_min;
    marker.points[16].z = t_W_M_a[0][2] + _z_min;
    marker.points[17].x = t_W_M_a[0][0] + _x_min;
    marker.points[17].y = t_W_M_a[0][1] + _y_min;
    marker.points[17].z = t_W_M_a[0][2] + _z_max;

    marker.points[18].x = t_W_M_a[0][0] + _x_max;
    marker.points[18].y = t_W_M_a[0][1] + _y_min;
    marker.points[18].z = t_W_M_a[0][2] + _z_min;
    marker.points[19].x = t_W_M_a[0][0] + _x_max;
    marker.points[19].y = t_W_M_a[0][1] + _y_min;
    marker.points[19].z = t_W_M_a[0][2] + _z_max;

    marker.points[20].x = t_W_M_a[0][0] + _x_max;
    marker.points[20].y = t_W_M_a[0][1] + _y_max;
    marker.points[20].z = t_W_M_a[0][2] + _z_min;
    marker.points[21].x = t_W_M_a[0][0] + _x_max;
    marker.points[21].y = t_W_M_a[0][1] + _y_max;
    marker.points[21].z = t_W_M_a[0][2] + _z_max;

    marker.points[22].x = t_W_M_a[0][0] + _x_min;
    marker.points[22].y = t_W_M_a[0][1] + _y_max;
    marker.points[22].z = t_W_M_a[0][2] + _z_min;
    marker.points[23].x = t_W_M_a[0][0] + _x_min;
    marker.points[23].y = t_W_M_a[0][1] + _y_max;
    marker.points[23].z = t_W_M_a[0][2] + _z_max;
  }
  return marker;
}

void export_to_ply(const torch::Tensor &_xyz, const std::string &output_path,
                   const std::string &_name) {
  std::string filename = output_path + _name + ".ply";

  tinyply::PlyFile mesh_ply;
  mesh_ply.add_properties_to_element(
      "vertex", {"x", "y", "z"}, tinyply::Type::FLOAT32, _xyz.size(0),
      reinterpret_cast<uint8_t *>(_xyz.data_ptr()), tinyply::Type::INVALID, 0);
  cout << "\nSaving inrange gt ply to: " << filename << "\n";
  std::filebuf fb_ascii;
  fb_ascii.open(filename, std::ios::out);
  std::ostream outstream_ascii(&fb_ascii);
  if (outstream_ascii.fail())
    throw std::runtime_error("failed to open " + filename);
  // Write an ASCII file
  mesh_ply.write(outstream_ascii, false);
  cout << "\n" << filename << " saved!\n";
}
} // namespace utils