import argparse
import json
import open3d as o3d

# gt_file = "/media/chrisliu/T7/RIM/RIM_exp_data/maicity/gt.ply"
# mesh_file = "/home/chrisliu/neural_slam_ws/src/neural_slam/neural_slam/output/mesh.ply"

# # Maicity
# # mesh_file = "/media/nrosliu/T7/RIM/RIM_exp_data/maicity/rim/eikonal/wo_ba.ply"

# # Replica
# # mesh_file = "/media/nrosliu/T7/RIM/RIM_exp_data/replica/office2/isdf.ply"
# # gt
# # mesh_file = "/media/nrosliu/T7/RIM/RIM_exp_data/replica/office2_mesh_vis.ply"


# Kitti
# o3d.visualization.draw_geometries([mesh_pred])

# MaiCity
# o3d.visualization.draw_geometries([mesh_pred], zoom=0.03,
#                                   front=[ -0.72190426320538992, -0.58953854009005113, 0.36235141025581019 ],
#                                   lookat=[ 32.067852271634891, 2.4683601677175688, -0.69043952969561273 ],
#                                   up=[ 0.35904725551067312, 0.1285241564800827, 0.92442772000375284 ])

# Replica

# # office-0
# o3d.visualization.draw_geometries([mesh_pred], zoom=0.7,
#                                   front=[ 0.0, 0.0, 1.0 ],
#                                   lookat=[ 0.23547999999999991, -0.68000000000000016, 0.30000000000000004 ],
#                                   up=[ -0.99556611441386555, -0.094064402570175856, 0.0 ])

# office-1
# o3d.visualization.draw_geometries([mesh_pred], zoom=0.7,
#                                   front=[ 0.0, 0.0, 1.0 ],
#                                   lookat=[ 0.39675066599231196, 0.90011347422962273, 0.30000000000000004 ],
#                                   up=[ -0.56958429383843612, 0.82193292439990362, 0.0 ])

# # office-2
# o3d.visualization.draw_geometries([mesh_pred], zoom=0.53999999999999981,
#                                   front=[ 0.0, 0.0, 1.0 ],
#                                   lookat=[  -0.19146258738045083, 1.3229079882497343, 0.30000000000000004 ],
#                                   up=[ -0.91357908907037211, 0.40666109724604815, 0.0  ])
# # sofa view
# o3d.visualization.draw_geometries([mesh_pred], zoom=0.15999999999999961,
#                                   front=[ 0.0043101294002288763, 0.8598465790717631, 0.51053431152385798 ],
#                                   lookat=[ -0.076902257737979235, -0.18914642945347884, -0.8837722163448466 ],
#                                   up=[ 0.081179347037609456, -0.50915485375310721, 0.85683793596817337 ])

# # office-3
# o3d.visualization.draw_geometries([mesh_pred], zoom=0.53999999999999981,
#                                   front=[ 0.0, 0.0, 1.0 ],
#                                   lookat=[   -0.77914147233535458, -1.0741181053608653, 0.30000000000000004 ],
#                                   up=[ -0.78704244721711858, 0.61689884606675971, 0.0  ])

# # office-4
# o3d.visualization.draw_geometries([mesh_pred], zoom=0.65999999999999992,
#                                   front=[ 0.0, 0.0, 1.0 ],
#                                   lookat=[ 2.1268790597737675, 0.80073796303055078, 0.30000000000000004 ],
#                                   up=[0.99461799587402977, 0.10361004914364463, 0.0  ])

# room-1
# o3d.visualization.draw_geometries([mesh_pred], zoom=0.65999999999999992,
#                                   front=[ 0.0, 0.0, 1.0 ],
#                                   lookat=[ -1.5812252783406415, -0.10107216502364728, 0.30000000000000004],
#                                   up=[-0.44036035495318687, 0.89782111680753396, 0.0])
# # bed view
# o3d.visualization.draw_geometries([mesh_pred], zoom=0.15999999999999961,
#                                   front=[ -0.57198424675854542, -0.66811409911883146, 0.47587558460031537 ],
#                                   lookat=[ -2.2267040323323948, 0.72790570837227198, -0.46032773885855038],
#                                   up=[0.23325419222826355, 0.42371633340573778, 0.87524679411760098])


# # room-2
# o3d.visualization.draw_geometries([mesh_pred], zoom=0.5199999999999998,
#                                   front=[ 0.13814865006261104, -0.041705247864472453, 0.98953303269089565 ],
#                                   lookat=[ 3.0622194275956858, -0.75240786313425012, 0.30000000000000004 ],
#                                   up=[0.20103302959736552, 0.97949530819283814, 0.013215984228272137])

# HITSZ office
# o3d.visualization.draw_geometries([mesh_pred], zoom=0.099999999999999617,
#                                   front=[ 0.81215957003889894, -0.4071705081225343, 0.41786242964577242 ],
#                                   lookat=[ 5.2368925575469403, 2.6278441652453939, 0.3843312598306059 ],
#                                   up=[-0.36626967716124464, 0.2016686425786087, 0.90839214119932765])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh_file", help="folder containing the rgb images")
    parser.add_argument("--view_config_file", default=None)
    args = parser.parse_args()

    print("Load the mesh from:", args.mesh_file)
    print("Load view config from:", args.view_config_file)

    mesh_o3d = o3d.io.read_triangle_mesh(
        args.mesh_file,
    )
    mesh_o3d.compute_vertex_normals()

    if args.view_config_file is None:
        o3d.visualization.draw_geometries([mesh_o3d])
    else:
        # Load JSON data from file
        with open(args.view_config_file, "r") as file:
            view_config = file.read()
        # Parse the JSON data
        view_config = json.loads(view_config)
        o3d.visualization.draw_geometries(
            [mesh_o3d],
            front=view_config["trajectory"][0]["front"],
            lookat=view_config["trajectory"][0]["lookat"],
            up=view_config["trajectory"][0]["up"],
            zoom=view_config["trajectory"][0]["zoom"],
            point_show_normal=True,
        )
