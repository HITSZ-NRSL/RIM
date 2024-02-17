import argparse
from eval_utils import eval_mesh

########################################### MaiCity Dataset ###########################################
dataset_name = "maicity_01_"

# ground truth point cloud (or mesh) file
# (optional masked by the intersection part of all the compared method)

# MaiCity
gt_pcd_path = "/media/chrisliu/T7/RIM/RIM_exp_data/maicity/gt.ply"
# pred_mesh_path = "/media/nrosliu/T7/RIM/RIM_exp_data/maicity/vdbfusion/maicity_01_100_scans.ply"
pred_mesh_path = "/media/chrisliu/T7/RIM/RIM_exp_data/maicity/rim/eikonal/50l2.ply"

# # Replica
# gt_pcd_path = "/media/nrosliu/T7/RIM/RIM_exp_data/replica/room2_mesh.ply"
# pred_mesh_path = "/media/nrosliu/T7/RIM/RIM_exp_data/replica/room2/vdb.ply"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt_pcd", help="folder containing the depth images")
    parser.add_argument("--pred_mesh", help="folder containing the rgb images")
    # evaluation parameters
    parser.add_argument("--down_sample_vox", type=float, default=0.01)
    parser.add_argument("--dist_thre", type=float, default=0.1)
    parser.add_argument("--truncation_dist_acc", type=float, default=0.2)
    parser.add_argument("--truncation_dist_com", type=float, default=2.0)
    args = parser.parse_args()

    output_txt_path = args.pred_mesh.replace("mesh.ply", "structure_eval.txt")

    # evaluation
    eval_metric = eval_mesh(
        args.pred_mesh,
        args.gt_pcd,
        down_sample_res=args.down_sample_vox,
        threshold=args.dist_thre,
        truncation_acc=args.truncation_dist_acc,
        truncation_com=args.truncation_dist_com,
        gt_bbx_mask_on=True,
    )

    try:
        with open(output_txt_path, "w") as txtfile:
            for key in eval_metric:
                print(key + ": " + str(eval_metric[key]))
                txtfile.write(key + ": " + str(eval_metric[key]) + "\n")
            print(f"Structure evaluation results are written into {output_txt_path}")
    except IOError:
        print("I/O error")
