#pragma once
#include <torch/torch.h>

#include "mesh_msgs/MeshVertexColorsStamped.h"
#include "ros/publisher.h"
#include <mesh_msgs/MeshGeometryStamped.h>

extern torch::Tensor TRIANGLE_TABLE;

// index here represented in binary: 100->4
extern torch::Tensor EDGE_INDEX_PAIRS;

torch::Tensor cal_vertex_config(const torch::Tensor &voxel_vertex_sdf,
                                float isovalue = 0.0);


torch::Tensor marching_cube(torch::Tensor _grid_sdf, torch::Tensor _grid_xyz,
                            float _isovalue = 0.0);

void tensor_to_mesh(mesh_msgs::MeshGeometry &mesh,
                    mesh_msgs::MeshVertexColors &mesh_color,
                    const torch::Tensor &face_xyz,
                    const torch::Tensor &face_normal_color);

void pub_mesh(const ros::Publisher &_mesh_pub,
              const ros::Publisher &_mesh_color_pub,
              const torch::Tensor &_face_xyz, const std_msgs::Header &_header,
              const std::string &_uuid);

// stack neighbors in marching cubes rule.
torch::Tensor stack_neighbors(const torch::Tensor &_xyz);