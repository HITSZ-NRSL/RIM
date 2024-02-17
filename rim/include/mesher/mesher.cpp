#include "mesher.h"
#include "marching_cube/marching_cube.h"
#include "params/params.h"
#include "utils/utils.h"

using namespace std;

void face_to_ply(const torch::Tensor &face_xyz, const std::string &filename) {

  auto face_num = face_xyz.size(0);
  auto vertex_num = face_num * 3;

  torch::Tensor vertex_index;
  vertex_index = torch::arange(0, vertex_num).to(torch::kInt);

  tinyply::PlyFile mesh_ply;
  mesh_ply.add_properties_to_element(
      "vertex", {"x", "y", "z"}, tinyply::Type::FLOAT32, vertex_num,
      reinterpret_cast<uint8_t *>(face_xyz.data_ptr()), tinyply::Type::INVALID,
      0);

  if (k_save_normal) {
    auto tmp_face_normal_color = utils::cal_face_normal_color(face_xyz);
    tmp_face_normal_color =
        tmp_face_normal_color.unsqueeze(1).expand({face_num, 3, 3});
    tmp_face_normal_color = (tmp_face_normal_color * 255).to(torch::kUInt8);
    mesh_ply.add_properties_to_element(
        "vertex", {"red", "green", "blue"}, tinyply::Type::UINT8, vertex_num,
        reinterpret_cast<uint8_t *>(tmp_face_normal_color.data_ptr()),
        tinyply::Type::INVALID, 0);
  }

  mesh_ply.add_properties_to_element(
      "face", {"vertex_indices"}, tinyply::Type::INT32, face_num,
      reinterpret_cast<uint8_t *>(vertex_index.data_ptr()),
      tinyply::Type::UINT8, 3);
  printf("\033[1;34mSaving mesh to: %s\n\033[0m", filename.c_str());
  // make directory if not exist
  std::string dir = filename.substr(0, filename.find_last_of('/'));
  if (!dir.empty()) {
    std::filesystem::create_directories(dir);
  }
  std::filebuf fb_ascii;
  fb_ascii.open(filename, std::ios::out);
  std::ostream outstream_ascii(&fb_ascii);
  if (outstream_ascii.fail())
    throw std::runtime_error("failed to open " + filename);
  // Write an ASCII file
  mesh_ply.write(outstream_ascii, false);
}

torch::Tensor xyz_sdf_mask_to_face(torch::Tensor &_mesh_xyz,
                                   torch::Tensor &_mesh_sdf,
                                   torch::Tensor &_mesh_mask, long x_num,
                                   long y_num, long z_num, float _isovalue) {

  _mesh_mask = _mesh_mask.view({x_num, y_num, z_num});
  auto xyz_valid_idx =
      stack_neighbors(_mesh_mask).view({-1, 8}).all(1).nonzero().squeeze();

  _mesh_sdf = _mesh_sdf.view({x_num, y_num, z_num});
  auto xyz_vertex_sdf =
      stack_neighbors(_mesh_sdf).view({-1, 8}).index({xyz_valid_idx});

  _mesh_xyz = _mesh_xyz.view({x_num, y_num, z_num, 3});
  auto xyz_vertex =
      stack_neighbors(_mesh_xyz).view({-1, 8, 3}).index({xyz_valid_idx});

  /// [n,3,3]
  return marching_cube(xyz_vertex_sdf, xyz_vertex);
}

Mesher::Mesher(){};

void Mesher::save_mesh() {
  if (vec_face_xyz_.empty()) {
    cout << "\n"
         << "No mesh to save! Please first meshing using 'v' first!"
         << "\n";
    return;
  }

  cout << "\n"
       << "Mesh saving start..."
       << "\n";
  auto face_xyz = torch::cat({vec_face_xyz_}, 0);

  if (k_downsample_mesh_size > 0) {
    cout << "\n"
         << "Remove duplicate mesh with threshold: " << k_downsample_mesh_size
         << "m."
         << "\n";
    /// remove duplicate vertex(or downsampling), threshold 1cm^3
    face_xyz = (face_xyz / k_downsample_mesh_size).floor().to(torch::kLong);
    face_xyz =
        (get<0>(torch::unique_dim(face_xyz, 0, false))).to(torch::kFloat32);
    face_xyz = face_xyz * k_downsample_mesh_size;
  }

  //     /// 100*1024*1024/(4*3+1*3+4*4+1*10) ~= 3e6 // byte: 4*3: xyz, 1*3:
  //     /// color, 4*3: pre+vertex_index, 1*10: space(char)
  //     int vis_batch_face_num = 3e6 / 3;
  //     auto iter = face_xyz.size(0) / vis_batch_face_num + 1;
  // #pragma omp parallel for
  //     for (int i = 0; i < iter; ++i) {
  //       long start = i * vis_batch_face_num;
  //       long end = (i + 1) * vis_batch_face_num;
  //       if (i == iter - 1) {
  //         end = end > face_xyz.size(0) ? face_xyz.size(0) : end;
  //         if (end == start)
  //           continue;
  //       }

  //       auto tmp_face_xyz = face_xyz.index({torch::indexing::Slice(start,
  //       end)});

  //       std::string filename =
  //           output_path + "/mesh/mesh_" + to_string(i) + ".ply";
  //       face_to_ply(tmp_face_xyz, filename);
  //     }
  // name in now time
  std::string filename = k_output_path + "/mesh.ply";
  face_to_ply(face_xyz, filename);
  printf("\033[1;32mExport mesh saved to %s\n\033[0m", filename.c_str());
}
