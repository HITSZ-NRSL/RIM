#!/bin/bash
root_path=$1


# For NeuralRGBD dataset, set is_focal_file to True, and already_kitti_format_pose to False
# For Replica dataset,    set is_focal_file to False, and already_kitti_format_pose to True

# depth_type="depth_filtered"
depth_type="depth"

base_path=${root_path}/"kitchen"
command="python3 src/RIM/scripts/data_convert/rgbd_to_kitti_format.py
        --output_root ${base_path}_kitti_format
        --depth_img_folder ${base_path}/${depth_type}/
        --rgb_img_folder ${base_path}/images/
        --intrinsic_file ${base_path}/focal.txt
        --pose_file ${base_path}/poses.txt
        --is_focal_file True
        --already_kitti_format_pose False
        --vis_on False"

echo "Convert RGBD dataset ${base_path} to KITTI format"
eval $command
echo "Done."

base_path=${root_path}/"breakfast_room"
command="python3 src/RIM/scripts/data_convert/rgbd_to_kitti_format.py
        --output_root ${base_path}_kitti_format
        --depth_img_folder ${base_path}/${depth_type}/
        --rgb_img_folder ${base_path}/images/
        --intrinsic_file ${base_path}/focal.txt
        --pose_file ${base_path}/poses.txt
        --is_focal_file True
        --already_kitti_format_pose False
        --vis_on False"

echo "Convert RGBD dataset ${base_path} to KITTI format"
eval $command
echo "Done."

base_path=${root_path}/"complete_kitchen"
command="python3 src/RIM/scripts/data_convert/rgbd_to_kitti_format.py
        --output_root ${base_path}_kitti_format
        --depth_img_folder ${base_path}/${depth_type}/
        --rgb_img_folder ${base_path}/images/
        --intrinsic_file ${base_path}/focal.txt
        --pose_file ${base_path}/poses.txt
        --is_focal_file True
        --already_kitti_format_pose False
        --vis_on False"

echo "Convert RGBD dataset ${base_path} to KITTI format"
eval $command
echo "Done."

base_path=${root_path}/"green_room"
command="python3 src/RIM/scripts/data_convert/rgbd_to_kitti_format.py
        --output_root ${base_path}_kitti_format
        --depth_img_folder ${base_path}/${depth_type}/
        --rgb_img_folder ${base_path}/images/
        --intrinsic_file ${base_path}/focal.txt
        --pose_file ${base_path}/poses.txt
        --is_focal_file True
        --already_kitti_format_pose False
        --vis_on False"

echo "Convert RGBD dataset ${base_path} to KITTI format"
eval $command
echo "Done."

base_path=${root_path}/"grey_white_room"
command="python3 src/RIM/scripts/data_convert/rgbd_to_kitti_format.py
        --output_root ${base_path}_kitti_format
        --depth_img_folder ${base_path}/${depth_type}/
        --rgb_img_folder ${base_path}/images/
        --intrinsic_file ${base_path}/focal.txt
        --pose_file ${base_path}/poses.txt
        --is_focal_file True
        --already_kitti_format_pose False
        --vis_on False"

echo "Convert RGBD dataset ${base_path} to KITTI format"
eval $command
echo "Done."

base_path=${root_path}/"morning_apartment"
command="python3 src/RIM/scripts/data_convert/rgbd_to_kitti_format.py
        --output_root ${base_path}_kitti_format
        --depth_img_folder ${base_path}/${depth_type}/
        --rgb_img_folder ${base_path}/images/
        --intrinsic_file ${base_path}/focal.txt
        --pose_file ${base_path}/poses.txt
        --is_focal_file True
        --already_kitti_format_pose False
        --vis_on False"

echo "Convert RGBD dataset ${base_path} to KITTI format"
eval $command
echo "Done."

base_path=${root_path}/"staircase"
command="python3 src/RIM/scripts/data_convert/rgbd_to_kitti_format.py
        --output_root ${base_path}_kitti_format
        --depth_img_folder ${base_path}/${depth_type}/
        --rgb_img_folder ${base_path}/images/
        --intrinsic_file ${base_path}/focal.txt
        --pose_file ${base_path}/poses.txt
        --is_focal_file True
        --already_kitti_format_pose False
        --vis_on False"

echo "Convert RGBD dataset ${base_path} to KITTI format"
eval $command
echo "Done."

base_path=${root_path}/"thin_geometry"
command="python3 src/RIM/scripts/data_convert/rgbd_to_kitti_format.py
        --output_root ${base_path}_kitti_format
        --depth_img_folder ${base_path}/${depth_type}/
        --rgb_img_folder ${base_path}/images/
        --intrinsic_file ${base_path}/focal.txt
        --pose_file ${base_path}/poses.txt
        --is_focal_file True
        --already_kitti_format_pose False
        --vis_on False"

echo "Convert RGBD dataset ${base_path} to KITTI format"
eval $command
echo "Done."

base_path=${root_path}/"whiteroom"
command="python3 src/RIM/scripts/data_convert/rgbd_to_kitti_format.py
        --output_root ${base_path}_kitti_format
        --depth_img_folder ${base_path}/${depth_type}/
        --rgb_img_folder ${base_path}/images/
        --intrinsic_file ${base_path}/focal.txt
        --pose_file ${base_path}/poses.txt
        --is_focal_file True
        --already_kitti_format_pose False
        --vis_on False"

echo "Convert RGBD dataset ${base_path} to KITTI format"
eval $command
echo "Done."
