#include "neural_slam/neural_slam.h"
#include <ros/ros.h>

#define BACKWARD_HAS_DW 1
#include "backward.hpp"
namespace backward {
backward::SignalHandling sh;
}

int main(int argc, char **argv) {
  torch::manual_seed(0);
  torch::cuda::manual_seed_all(0);

  ros::init(argc, argv, "neural_slam");
  ros::NodeHandle nh("neural_slam");
  ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME,
                                 ros::console::levels::Info);
  if (argc < 2)
    ROS_ERROR("Usage: rosrun neural_slam neural_slam_node <config_path> "
              "<data_path>(optional)");

  std::string config_path = std::string(argv[1]);
  std::string data_path;
  if (argc > 2)
    data_path = std::string(argv[2]);

  NeuralSLAM neural_slam(nh, config_path, data_path);

  ros::spin();
  return 0;
}