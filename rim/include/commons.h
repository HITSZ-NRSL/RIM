#pragma once

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <torch/torch.h>

typedef pcl::PointXYZ PointT;
typedef pcl::PointCloud<PointT> PointCloudT;

struct RaySamples {
  torch::Tensor xyz;
  torch::Tensor direction;
  torch::Tensor ray_sdf;

  torch::Tensor pred_sdf;
  torch::Tensor pred_gradient;
  torch::Tensor pred_normal;
};