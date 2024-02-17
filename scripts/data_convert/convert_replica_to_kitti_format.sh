#!/bin/bash
root_path=$1

# For NeuralRGBD dataset, set is_focal_file to True, and already_kitti_format_pose to False
# For Replica dataset,    set is_focal_file to False, and already_kitti_format_pose to True

base_path=${root_path}/"office0"
command="python3 src/RIM/scripts/data_convert/replica_to_kitti_format.py
        --output_root ${base_path}_kitti_format
        --depth_img_folder ${base_path}/results/
        --rgb_img_folder ${base_path}/results/
        --intrinsic_file ${root_path}/cam_params.json
        --pose_file ${base_path}/traj.txt
        --is_focal_file False
        --already_kitti_format_pose True
        --vis_on False"

echo "Convert RGBD dataset ${base_path} to KITTI format"
eval $command
echo "Done."


base_path=${root_path}/"office1"
command="python3 src/RIM/scripts/data_convert/replica_to_kitti_format.py
        --output_root ${base_path}_kitti_format
        --depth_img_folder ${base_path}/results/
        --rgb_img_folder ${base_path}/results/
        --intrinsic_file ${root_path}/cam_params.json
        --pose_file ${base_path}/traj.txt
        --is_focal_file False
        --already_kitti_format_pose True
        --vis_on False"

echo "Convert RGBD dataset ${base_path} to KITTI format"
eval $command
echo "Done."


base_path=${root_path}/"office2"
command="python3 src/RIM/scripts/data_convert/replica_to_kitti_format.py
        --output_root ${base_path}_kitti_format
        --depth_img_folder ${base_path}/results/
        --rgb_img_folder ${base_path}/results/
        --intrinsic_file ${root_path}/cam_params.json
        --pose_file ${base_path}/traj.txt
        --is_focal_file False
        --already_kitti_format_pose True
        --vis_on False"

echo "Convert RGBD dataset ${base_path} to KITTI format"
eval $command
echo "Done."


base_path=${root_path}/"office3"
command="python3 src/RIM/scripts/data_convert/replica_to_kitti_format.py
        --output_root ${base_path}_kitti_format
        --depth_img_folder ${base_path}/results/
        --rgb_img_folder ${base_path}/results/
        --intrinsic_file ${root_path}/cam_params.json
        --pose_file ${base_path}/traj.txt
        --is_focal_file False
        --already_kitti_format_pose True
        --vis_on False"

echo "Convert RGBD dataset ${base_path} to KITTI format"
eval $command
echo "Done."


base_path=${root_path}/"office4"
command="python3 src/RIM/scripts/data_convert/replica_to_kitti_format.py
        --output_root ${base_path}_kitti_format
        --depth_img_folder ${base_path}/results/
        --rgb_img_folder ${base_path}/results/
        --intrinsic_file ${root_path}/cam_params.json
        --pose_file ${base_path}/traj.txt
        --is_focal_file False
        --already_kitti_format_pose True
        --vis_on False"

echo "Convert RGBD dataset ${base_path} to KITTI format"
eval $command
echo "Done."


base_path=${root_path}/"room0"
command="python3 src/RIM/scripts/data_convert/replica_to_kitti_format.py
        --output_root ${base_path}_kitti_format
        --depth_img_folder ${base_path}/results/
        --rgb_img_folder ${base_path}/results/
        --intrinsic_file ${root_path}/cam_params.json
        --pose_file ${base_path}/traj.txt
        --is_focal_file False
        --already_kitti_format_pose True
        --vis_on False"

echo "Convert RGBD dataset ${base_path} to KITTI format"
eval $command
echo "Done."

base_path=${root_path}/"room1"
command="python3 src/RIM/scripts/data_convert/replica_to_kitti_format.py
        --output_root ${base_path}_kitti_format
        --depth_img_folder ${base_path}/results/
        --rgb_img_folder ${base_path}/results/
        --intrinsic_file ${root_path}/cam_params.json
        --pose_file ${base_path}/traj.txt
        --is_focal_file False
        --already_kitti_format_pose True
        --vis_on False"

echo "Convert RGBD dataset ${base_path} to KITTI format"
eval $command
echo "Done."

base_path=${root_path}/"room2"
command="python3 src/RIM/scripts/data_convert/replica_to_kitti_format.py
        --output_root ${base_path}_kitti_format
        --depth_img_folder ${base_path}/results/
        --rgb_img_folder ${base_path}/results/
        --intrinsic_file ${root_path}/cam_params.json
        --pose_file ${base_path}/traj.txt
        --is_focal_file False
        --already_kitti_format_pose True
        --vis_on False"

echo "Convert RGBD dataset ${base_path} to KITTI format"
eval $command
echo "Done."