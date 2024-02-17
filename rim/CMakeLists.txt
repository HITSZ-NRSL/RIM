cmake_minimum_required(VERSION 2.8.3)
project(neural_slam)

add_compile_options(-std=c++17)
set(CMAKE_BUILD_TYPE "RelWithDebInfo")
# set(CMAKE_BUILD_TYPE "Release") set(CMAKE_BUILD_TYPE "Debug")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")

find_package(
  catkin REQUIRED
  COMPONENTS roscpp
             roslib
             std_msgs
             geometry_msgs
             nav_msgs
             mesh_msgs
             cv_bridge
             tf)

find_package(OpenMP REQUIRED)
if(OPENMP_FOUND)
  message("OPENMP FOUND")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
message(Eigen: ${EIGEN3_INCLUDE_DIR})

set(CMAKE_SUPPRESS_DEVELOPER_WARNINGS
    1
    CACHE INTERNAL "No dev warnings") # 关闭pcl烦人的警告
find_package(PCL REQUIRED)

find_package(GTest REQUIRED) # 查找gtest包

# 指定libTorch位置
set(Torch_DIR "./thirdparty/libtorch/share/cmake/Torch")
find_package(Torch REQUIRED)

include_directories(${catkin_INCLUDE_DIRS} ${EIGEN3_INCLUDE_DIR}
                    ${PCL_INCLUDE_DIRS} ${GTEST_INCLUDE_DIRS} include)

catkin_package(
  # CATKIN_DEPENDS geometry_msgs nav_msgs roscpp rospy std_msgs message_runtime
  # DEPENDS EIGEN3 PCL INCLUDE_DIRS
)

add_library(
  neural_slam_lib
  include/neural_slam/neural_slam.cpp
  include/params/params.cpp
  include/marching_cube/marching_cube.cpp
  include/utils/utils.cpp
  include/llog/llog.cpp
  include/mesher/mesher.cpp
  include/neural_net/sub_map.cpp
  include/neural_net/local_map.cpp
  include/neural_net/global_map.cpp)
target_link_libraries(neural_slam_lib ${catkin_LIBRARIES} ${OpenCV_LIBS}
                      ${PCL_LIBRARIES} ${TORCH_LIBRARIES} dw)

add_executable(neural_slam_node src/neural_slam_node.cpp)
target_link_libraries(neural_slam_node neural_slam_lib)

