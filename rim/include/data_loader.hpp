#pragma once
#include <dirent.h>
#include <fstream>
#include <iostream>
#include <memory>

#include <opencv2/core/core.hpp>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/point_types.h>
#include <tinyply/tinyply.h>
#include <torch/torch.h>

#include "commons.h"

struct vec3uc {
  uchar r, g, b;
};

void read_filelists(const std::string &dir_path,
                    std::vector<std::string> &out_filelsits) {
  struct dirent *ptr;
  DIR *dir;
  dir = opendir(dir_path.c_str());
  out_filelsits.clear();
  while ((ptr = readdir(dir)) != nullptr) {
    std::string tmp_file = ptr->d_name;
    if (tmp_file[0] == '.')
      continue;
    out_filelsits.emplace_back(ptr->d_name);
  }
}

bool computePairNum(std::string pair1, std::string pair2) {
  return pair1 < pair2;
}
void sort_filelists(std::vector<std::string> &filists) {
  if (filists.empty())
    return;

  std::sort(filists.begin(), filists.end(), computePairNum);
}

bool parseVectorOfFloats(const std::string &input, std::vector<float> *output) {
  output->clear();
  // Parse the line as a stringstream for space-delimeted doubles.
  std::stringstream line_stream(input);
  if (line_stream.eof()) {
    return false;
  }

  while (!line_stream.eof()) {
    std::string element;
    std::getline(line_stream, element, ' ');
    if (element.empty()) {
      continue;
    }
    try {
      output->emplace_back(std::stof(element));
    } catch (const std::exception &exception) {
      std::cout << "Could not parse number in import file.\n";
      return false;
    }
  }
  return true;
}

void read_ply_file(const std::string &filepath, PointCloudT &_points) {
  std::unique_ptr<std::istream> file_stream =
      std::make_unique<std::ifstream>(filepath, std::ios::binary);

  if (!file_stream || file_stream->fail())
    throw std::runtime_error("file_stream failed to open " + filepath);

  file_stream->seekg(0, std::ios::end);
  const float size_mb = file_stream->tellg() * float(1e-6);
  file_stream->seekg(0, std::ios::beg);

  tinyply::PlyFile file;
  file.parse_header(*file_stream);

  // Because most people have their own mesh types, tinyply treats parsed data
  // as structured/typed byte buffers. See examples below on how to marry your
  // own application-specific data structures with this one.
  std::shared_ptr<tinyply::PlyData> vertices, colors;

  // The header information can be used to programmatically extract properties
  // on elements known to exist in the header prior to reading the data. For
  // brevity of this sample, properties like vertex position are hard-coded:
  try {
    vertices = file.request_properties_from_element("vertex", {"x", "y", "z"});
  } catch (const std::exception &e) {
    std::cerr << "tinyply exception: " << e.what() << "\n";
  }

  file.read(*file_stream);

  _points.clear();
  _points.resize(vertices->count);

  if (vertices->t == tinyply::Type::FLOAT32) {
    std::vector<Eigen::Vector3f> verts(vertices->count);
    std::memcpy(verts.data(), vertices->buffer.get(),
                vertices->buffer.size_bytes());

#pragma omp parallel for
    for (size_t i = 0; i < vertices->count; i++) {
      _points[i].x = verts[i].x();
      _points[i].y = verts[i].y();
      _points[i].z = verts[i].z();
    }
  }
  if (vertices->t == tinyply::Type::FLOAT64) {
    std::vector<Eigen::Vector3d> verts(vertices->count);
    std::memcpy(verts.data(), vertices->buffer.get(),
                vertices->buffer.size_bytes());

#pragma omp parallel for
    for (size_t i = 0; i < vertices->count; i++) {
      _points[i].x = verts[i].x();
      _points[i].y = verts[i].y();
      _points[i].z = verts[i].z();
    }
  }
}

void read_ply_file_to_tensor(const std::string &filepath,
                             torch::Tensor &_points, torch::Device &_device) {
  std::unique_ptr<std::istream> file_stream =
      std::make_unique<std::ifstream>(filepath, std::ios::binary);

  if (!file_stream || file_stream->fail())
    throw std::runtime_error("file_stream failed to open " + filepath);

  file_stream->seekg(0, std::ios::end);
  const float size_mb = file_stream->tellg() * float(1e-6);
  file_stream->seekg(0, std::ios::beg);

  tinyply::PlyFile file;
  file.parse_header(*file_stream);

  // Because most people have their own mesh types, tinyply treats parsed data
  // as structured/typed byte buffers. See examples below on how to marry your
  // own application-specific data structures with this one.
  std::shared_ptr<tinyply::PlyData> vertices, colors;

  // The header information can be used to programmatically extract properties
  // on elements known to exist in the header prior to reading the data. For
  // brevity of this sample, properties like vertex position are hard-coded:
  try {
    vertices = file.request_properties_from_element("vertex", {"x", "y", "z"});
  } catch (const std::exception &e) {
    std::cerr << "tinyply exception: " << e.what() << "\n";
  }

  file.read(*file_stream);

  if (vertices->t == tinyply::Type::FLOAT32) {
    _points = torch::from_blob(vertices->buffer.get(),
                               {(long)vertices->count, 3}, torch::kFloat32)
                  .to(_device);
  }
  if (vertices->t == tinyply::Type::FLOAT64) {
    _points = torch::from_blob(vertices->buffer.get(),
                               {(long)vertices->count, 3}, torch::kFloat64)
                  .to(_device)
                  .to(torch::kFloat32);
  }
}

void read_ply_file_to_tensor(const std::string &filepath,
                             torch::Tensor &_points, torch::Tensor &_colors,
                             torch::Device &_device) {
  std::unique_ptr<std::istream> file_stream =
      std::make_unique<std::ifstream>(filepath, std::ios::binary);

  if (!file_stream || file_stream->fail())
    throw std::runtime_error("file_stream failed to open " + filepath);

  file_stream->seekg(0, std::ios::end);
  const float size_mb = file_stream->tellg() * float(1e-6);
  file_stream->seekg(0, std::ios::beg);

  tinyply::PlyFile file;
  file.parse_header(*file_stream);

  // Because most people have their own mesh types, tinyply treats parsed data
  // as structured/typed byte buffers. See examples below on how to marry your
  // own application-specific data structures with this one.
  std::shared_ptr<tinyply::PlyData> vertices, colors;

  // The header information can be used to programmatically extract properties
  // on elements known to exist in the header prior to reading the data. For
  // brevity of this sample, properties like vertex position are hard-coded:
  try {
    vertices = file.request_properties_from_element("vertex", {"x", "y", "z"});
  } catch (const std::exception &e) {
    std::cerr << "tinyply exception: " << e.what() << "\n";
  }

  // try {
  //   colors = file.request_properties_from_element("vertex",
  //                                                 {"red", "green", "blue"});
  // } catch (const std::exception &e) {
  //   std::cerr << "tinyply exception: " << e.what() << "\n";
  // }

  file.read(*file_stream);

  if (vertices->t == tinyply::Type::FLOAT32) {
    _points = torch::from_blob(vertices->buffer.get(),
                               {(long)vertices->count, 3}, torch::kFloat32)
                  .to(_device);
  }
  if (vertices->t == tinyply::Type::FLOAT64) {
    _points = torch::from_blob(vertices->buffer.get(),
                               {(long)vertices->count, 3}, torch::kFloat64)
                  .to(_device)
                  .to(torch::kFloat32);
  }
  // _colors = torch::from_blob(colors->buffer.get(), {(long)colors->count, 3},
  //                            torch::kUInt8)
  //               .to(_device);
}

template <typename VectorType>
std::pair<std::vector<VectorType>, std::vector<vec3uc>>
read_ply_file(const std::string &filepath) {
  std::unique_ptr<std::istream> file_stream =
      std::make_unique<std::ifstream>(filepath, std::ios::binary);

  if (!file_stream || file_stream->fail())
    throw std::runtime_error("file_stream failed to open " + filepath);

  file_stream->seekg(0, std::ios::end);
  const float size_mb = file_stream->tellg() * float(1e-6);
  file_stream->seekg(0, std::ios::beg);

  tinyply::PlyFile file;
  file.parse_header(*file_stream);

  // Because most people have their own mesh types, tinyply treats parsed data
  // as structured/typed byte buffers. See examples below on how to marry your
  // own application-specific data structures with this one.
  std::shared_ptr<tinyply::PlyData> vertices, colors;

  // The header information can be used to programmatically extract properties
  // on elements known to exist in the header prior to reading the data. For
  // brevity of this sample, properties like vertex position are hard-coded:
  try {
    vertices = file.request_properties_from_element("vertex", {"x", "y", "z"});
  } catch (const std::exception &e) {
    std::cerr << "tinyply exception: " << e.what() << "\n";
  }

  try {
    colors = file.request_properties_from_element("vertex",
                                                  {"red", "green", "blue"});
  } catch (const std::exception &e) {
    std::cerr << "tinyply exception: " << e.what() << "\n";
  }
  file.read(*file_stream);

  std::vector<vec3uc> cols(vertices->count);
  std::memcpy(cols.data(), colors->buffer.get(), colors->buffer.size_bytes());
  if (vertices->t == tinyply::Type::FLOAT32) {
    std::vector<Eigen::Vector3f> verts(vertices->count);
    std::memcpy(verts.data(), vertices->buffer.get(),
                vertices->buffer.size_bytes());
    return std::make_pair(verts, cols);
  }
  if (vertices->t == tinyply::Type::FLOAT64) {
    std::vector<Eigen::Vector3d> verts(vertices->count);
    std::memcpy(verts.data(), vertices->buffer.get(),
                vertices->buffer.size_bytes());

    return std::make_pair(verts, cols);
  }
}

class DataLoader {
public:
  DataLoader(const std::string &calib_file, const std::string &poses_file,
             const std::string &dataset_path)
      : calib_path_(calib_file), poses_path_(poses_file),
        dataset_path_(dataset_path) {
    load_calib();
    load_poses();
    load_depth_list();
  }

  bool get_next_data(int idx, torch::Tensor &_pose, PointCloudT &_points) {
    if (idx >= depth_name_vec_.size()) {
      std::cout << "End of the data!\n";
      return false;
    }

    std::cout << "\nData idx: " << idx
              << ", Depth file:" << depth_name_vec_[idx] << "\n";
    _pose = poses_vec_[idx];

    std::string infile = dataset_path_ + "/" + depth_name_vec_[idx];

    if (infile.find(".bin") != std::string::npos) {
      std::ifstream input(infile.c_str(), std::ios::in | std::ios::binary);
      if (!input) {
        std::cerr << "Could not read file: " << infile << "\n";
        return false;
      }

      const size_t kMaxNumberOfPoints = 1e6; // From the Readme of raw files.
      _points.clear();
      _points.reserve(kMaxNumberOfPoints);

      while (input.is_open() && !input.eof()) {
        PointT point;

        input.read((char *)&point.x, 3 * sizeof(float));
        // pcl::PointXYZI
        float intensity;
        input.read((char *)&intensity, sizeof(float));
        // input.read((char *)&point.intensity, sizeof(float));
        _points.push_back(point);
      }
      input.close();
    } else if (infile.find(".ply") != std::string::npos) {
      read_ply_file(infile, _points);
    } else if (infile.find(".pcd") != std::string::npos) {
      pcl::io::loadPCDFile<PointT>(infile, _points);
    }

    return true;
  }

  bool get_next_data(int idx, torch::Tensor &_pose, torch::Tensor &_points,
                     torch::Tensor &_colors, torch::Device &_device) {
    if (idx >= depth_name_vec_.size()) {
      std::cout << "End of the data!\n";
      return false;
    }

    std::cout << "\nData idx: " << idx
              << ", Depth file:" << depth_name_vec_[idx] << "\n";
    _pose = poses_vec_[idx];

    std::string infile = dataset_path_ + "/" + depth_name_vec_[idx];

    if (infile.find(".ply") != std::string::npos) {
      read_ply_file_to_tensor(infile, _points, _colors, _device);
    }

    return true;
  }

private:
  // Base paths.
  std::string calib_path_, poses_path_;
  std::string dataset_path_;

  torch::Tensor Tr;

  std::vector<torch::Tensor> poses_vec_;
  std::vector<std::string> depth_name_vec_;

  bool load_calib() {
    if (calib_path_.empty()) {
      Tr = torch::eye(4);
    } else {
      std::ifstream import_file(calib_path_, std::ios::in);
      if (!import_file) {
        std::cerr << "Could not open calibration file: " << calib_path_ << "\n";
        return false;
      }

      std::string line;
      while (std::getline(import_file, line)) {
        std::stringstream line_stream(line);

        // Check what the header is. Each line consists of two parts:
        // a header followed by a ':' followed by space-separated data.
        std::string header;
        std::getline(line_stream, header, ':');
        std::string data;
        std::getline(line_stream, data, ':');

        std::vector<float> parsed_floats;
        if (header == "Tr") {
          // Parse the translation matrix.
          if (parseVectorOfFloats(data, &parsed_floats)) {
            Tr =
                torch::from_blob(parsed_floats.data(), {3, 4}, torch::kFloat32);
            Tr = torch::cat({Tr, torch::tensor({{0, 0, 0, 1}})}, 0);
          }
        }
      }
    }
    std::cout << "Tr:"
              << "\n"
              << Tr << "\n";

    return true;
  }
  bool load_poses() {
    std::ifstream import_file(poses_path_, std::ios::in);
    if (!import_file) {
      std::cerr << "Could not open poses file: " << poses_path_ << "\n";
      return false;
    }

    auto Tr_inv = Tr.inverse();
    std::string line;
    while (std::getline(import_file, line)) {
      std::vector<float> parsed_floats;
      if (parseVectorOfFloats(line, &parsed_floats)) {
        auto pose =
            torch::from_blob(parsed_floats.data(), {3, 4}, torch::kFloat);
        pose = torch::cat({pose, torch::tensor({{0, 0, 0, 1}})}, 0);
        // https://github.com/PRBonn/semantic-kitti-api/issues/115
        pose = Tr_inv.mm(pose.mm(Tr));
        poses_vec_.emplace_back(pose.slice(0, 0, 3));
      }
    }
    std::cout << "Load " << poses_vec_.size() << " poses."
              << "\n";
    return true;
  }

  bool load_depth_list() {
    read_filelists(dataset_path_, depth_name_vec_);
    sort_filelists(depth_name_vec_);
    // for (auto &name : depth_name_vec_) {
    //   std::cout << name << "\n";
    // }

    std::cout << "Load " << depth_name_vec_.size() << " depth data."
              << "\n";
    return true;
  }
};
