import os
import json
import torch
import numpy as np
from tqdm import tqdm
from .scannet_io import get_scannet_to_uniscenes
from typing import List, Tuple, Dict
import pickle

NYU_COLORS = np.array(
    [
        [255, 255, 255, 255],
        [214, 38, 41, 255],
        [43, 160, 4, 255],
        [159, 216, 229, 255],
        [114, 158, 206, 255],
        [204, 204, 90, 255],
        [254, 186, 119, 255],
        [147, 102, 188, 255],
        [30, 119, 181, 255],
        [160, 188, 32, 255],
        [253, 127, 13, 255],
        [196, 175, 214, 255],
        [0, 0, 0, 255],
    ]
).astype(np.uint8)

category_dict = {"ceiling": 1, "floor": 2, "wall": 3, "window": 4, "chair": 5, "bed": 6, "sofa": 7, "table": 8, "tv": 9, "furniture": 10, "objects": 11}

def parse_frames(frames: List[Dict]) -> Tuple[List[str], List[np.ndarray]]:
    file_names = []
    camera_to_worlds = []
    for frame in frames:
        # Convert the poses from nerfstudio to colmap convention
        image_name = frame["file_path"]
        camera_to_world = np.array(frame["transform_matrix"])
        camera_to_world[2, :] *= -1
        camera_to_world = camera_to_world[np.array([1, 0, 2, 3]), :]
        camera_to_world[0:3, 1:3] *= -1
        file_names.append(image_name)
        camera_to_worlds.append(camera_to_world)
    return file_names, camera_to_worlds


# def init_scannet(args, vis=False):
#     # Load camera intrinsics
#     data_root = cfg.data.source_path
#     scene_id = args.scene_id

#     cam_info_path = os.path.join(data_root, "data", scene_id, "dslr", "nerfstudio", "transforms_undistorted.json")
#     with open(cam_info_path, "r") as f:
#         cam_info = json.load(f)
#     fx, fy, cx, cy = cam_info["fl_x"], cam_info["fl_y"], cam_info["cx"], cam_info["cy"]
#     org_intrinsic = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
#     org_height, org_width = cam_info["h"], cam_info["w"]
#     width, height = 480, 480
#     cam_intrinsic = org_intrinsic.copy()
#     cam_intrinsic[0, :] *= width / org_width
#     cam_intrinsic[1, :] *= height / org_height

#     train_frames = cam_info["frames"]
#     test_frames = cam_info["test_frames"]

#     train_file_names, train_camera_to_worlds = parse_frames(train_frames)
#     test_file_names, test_camera_to_worlds = parse_frames(test_frames)

#     pcd_path = os.path.join(data_root, "data", scene_id, "scans", "mesh_aligned_0.05.ply")
#     pcd = o3d.io.read_point_cloud(pcd_path)

#     if vis:
#         o3d_cam_axis_list = []
#         for idx in range(len(train_camera_to_worlds)):
#             o3d_axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0,0,0])
#             o3d_axis.transform(train_camera_to_worlds[idx])
#             o3d_cam_axis_list.append(o3d_axis)
#         o3d.visualization.draw_geometries([pcd, *o3d_cam_axis_list])

#     return train_file_names, train_camera_to_worlds, test_file_names, test_camera_to_worlds, cam_intrinsic, org_height, org_width, width, height, pcd

# Scannet++ / Uniscenes mapping dict
split_dict_path = "split.json"
train_scene_names = json.load(open(split_dict_path, "r"))["train"]
val_scene_names = json.load(open(split_dict_path, "r"))["val"]
print(f"Found {len(train_scene_names)} train scenes and {len(val_scene_names)} val scenes.")
mapping_dict = "scannetpp_to_uniscenes.json"
if os.path.exists(mapping_dict):
    scannet_to_uniscenes = json.load(open(mapping_dict, "r"))
uniscenes_to_scannet = {v: k for k, v in scannet_to_uniscenes.items()}
train_scene_ids = [uniscenes_to_scannet[n] for n in train_scene_names if n in uniscenes_to_scannet]
val_scene_ids = [uniscenes_to_scannet[n] for n in val_scene_names if n in uniscenes_to_scannet]
print(f"Using {len(train_scene_ids)} train scenes and {len(val_scene_ids)} val scenes after mapping.")

data_root_dir = "/media/sequor/PortableSSD/scannetpp"
pointcloud_dir = os.path.join(data_root_dir, "pointcloud")
os.makedirs(os.path.join(data_root_dir, "occupancy"), exist_ok=True)


def generate_scannetpp_info(data_root_dir, scene_id_list, split="train"):

    data_list = []

    for scene_id in tqdm(scene_id_list):
    
        scene_data_dict = {}

        occupancy_path = os.path.join(data_root_dir, "occupancy", scene_id, "occupancy.npy")
        axis_align_matrix_path = os.path.join(data_root_dir, "occupancy", scene_id, "axis_align_matrix.npy")
        if os.path.exists(occupancy_path) and os.path.exists(axis_align_matrix_path):
            AssertionError(f"Occupancy data already exists for scene {scene_id} in split {split}.")
        axis_align_matrix = np.load(axis_align_matrix_path)

        scene_data_dict["sample_idx"] = scene_id
        scene_data_dict["axis_align_matrix"] = axis_align_matrix

        cam_info_path = os.path.join(data_root_dir, "data", scene_id, "dslr", "nerfstudio", "transforms_undistorted.json")
        with open(cam_info_path, "r") as f:
            cam_info = json.load(f)
        fx, fy, cx, cy = cam_info["fl_x"], cam_info["fl_y"], cam_info["cx"], cam_info["cy"]
        org_intrinsic = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        org_height, org_width = cam_info["h"], cam_info["w"]

        cam2img = np.eye(4)
        cam2img[:3, :3] = org_intrinsic
        depth_cam2img = cam2img.copy()

        scene_data_dict["cam2img"] = cam2img.astype(np.float32)
        scene_data_dict["depth_cam2img"] = depth_cam2img.astype(np.float32)

        train_frames = cam_info["frames"]
        test_frames = cam_info["test_frames"]

        train_file_names, train_camera_to_worlds = parse_frames(train_frames)
        test_file_names, test_camera_to_worlds = parse_frames(test_frames)

        scene_image_list = []

        for i in range(len(train_file_names)):
            image_dict = {}
            image_dict["img_path"] = os.path.join("data", scene_id, "dslr", "undistorted_images", train_file_names[i])
            image_dict["cam2global"] = train_camera_to_worlds[i].astype(np.float32)
            image_dict["depth_path"] = os.path.join("render_orgpose", scene_id, "dslr", "render_depth", train_file_names[i].replace("JPG", "png"))
            scene_image_list.append(image_dict)

        scene_data_dict["images"] = scene_image_list
        
        data_list.append(scene_data_dict)

    return data_list

metadata = {"dataset": "scannetpp", "categories": category_dict}
train_data_list = generate_scannetpp_info(data_root_dir, train_scene_ids, split="train")
train_info_save_path = os.path.join(data_root_dir, f"scannetpp_infos_train.pkl")
train_info_dict = {"metainfo": metadata, "data_list": train_data_list}
with open(train_info_save_path, "wb") as f:
    pickle.dump(train_info_dict, f)
val_data_list = generate_scannetpp_info(data_root_dir, val_scene_ids, split="val")
val_info_save_path = os.path.join(data_root_dir, f"scannetpp_infos_val.pkl")
val_info_dict = {"metainfo": metadata, "data_list": val_data_list}
with open(val_info_save_path, "wb") as f:
    pickle.dump(val_info_dict, f)
#compute_scene_extent(data_root_dir, pointcloud_dir, scannetpp_to_nyu, val_scene_ids, split="val")


# ========== data_list[0] ==========
# keys: ['axis_align_matrix', 'cam2img', 'depth_cam2img', 'images', 'instances', 'sample_idx']
# sample_idx: scannet/scene0191_00
# cam2img: {'type': "<class 'numpy.ndarray'>", 'dtype': 'float32', 'shape': [4, 4], 'flat_head': array([[1.170188e+03, 0.000000e+00, 6.477500e+02, 0.000000e+00],
#        [0.000000e+00, 1.170188e+03, 4.837500e+02, 0.000000e+00],
#        [0.000000e+00, 0.000000e+00, 1.000000e+00, 0.000000e+00],
#        [0.000000e+00, 0.000000e+00, 0.000000e+00, 1.000000e+00]],
#       dtype=float32)}
# depth_cam2img: {'type': "<class 'numpy.ndarray'>", 'dtype': 'float32', 'shape': [4, 4], 'flat_head': array([[577.8706,   0.    , 319.5   ,   0.    ],
#        [  0.    , 577.8706, 239.5   ,   0.    ],
#        [  0.    ,   0.    ,   1.    ,   0.    ],
#        [  0.    ,   0.    ,   0.    ,   1.    ]], dtype=float32)}
# axis_align_matrix: {'type': "<class 'numpy.ndarray'>", 'dtype': 'float64', 'shape': [4, 4], 'flat_head': array([[-0.21644 ,  0.976296,  0.      , -1.01457 ],
#        [-0.976296, -0.21644 ,  0.      ,  3.91808 ],
#        [ 0.      ,  0.      ,  1.      , -0.070665],
#        [ 0.      ,  0.      ,  0.      ,  1.      ]])}
# images: list(len=109)
#   - images[0] keys: ['cam2global', 'depth_path', 'img_path', 'visible_instance_ids']
#       img_path: scannet/posed_images/scene0191_00/00000.jpg
#   - images[1] keys: ['cam2global', 'depth_path', 'img_path', 'visible_instance_ids']
#       img_path: scannet/posed_images/scene0191_00/00010.jpg
# instances: list(len=20)
#   - instances[0] keys: ['bbox_3d', 'bbox_id', 'bbox_label_3d']
#       bbox_3d: list -> 
#       bbox_label_3d: int -> 276
#       bbox_id: int -> 1
# occupancy.npy keys:  (2384, 4)
# scannet/posed_images/scene0191_00/00000.jpg
# (40, 40, 16)

# ========== data_list[1] ==========
# keys: ['axis_align_matrix', 'cam2img', 'depth_cam2img', 'images', 'instances', 'sample_idx']
# sample_idx: scannet/scene0191_01
# cam2img: {'type': "<class 'numpy.ndarray'>", 'dtype': 'float32', 'shape': [4, 4], 'flat_head': array([[1.170188e+03, 0.000000e+00, 6.477500e+02, 0.000000e+00],
#        [0.000000e+00, 1.170188e+03, 4.837500e+02, 0.000000e+00],
#        [0.000000e+00, 0.000000e+00, 1.000000e+00, 0.000000e+00],
#        [0.000000e+00, 0.000000e+00, 0.000000e+00, 1.000000e+00]],
#       dtype=float32)}
# depth_cam2img: {'type': "<class 'numpy.ndarray'>", 'dtype': 'float32', 'shape': [4, 4], 'flat_head': array([[577.8706,   0.    , 319.5   ,   0.    ],
#        [  0.    , 577.8706, 239.5   ,   0.    ],
#        [  0.    ,   0.    ,   1.    ,   0.    ],
#        [  0.    ,   0.    ,   0.    ,   1.    ]], dtype=float32)}
# axis_align_matrix: {'type': "<class 'numpy.ndarray'>", 'dtype': 'float64', 'shape': [4, 4], 'flat_head': array([[ 0.656059,  0.75471 ,  0.      , -4.79916 ],
#        [-0.75471 ,  0.656059,  0.      , -0.401519],
#        [ 0.      ,  0.      ,  1.      , -0.076288],
#        [ 0.      ,  0.      ,  0.      ,  1.      ]])}
# images: list(len=118)
#   - images[0] keys: ['cam2global', 'depth_path', 'img_path', 'visible_instance_ids']
#       img_path: scannet/posed_images/scene0191_01/00000.jpg
#   - images[1] keys: ['cam2global', 'depth_path', 'img_path', 'visible_instance_ids']
#       img_path: scannet/posed_images/scene0191_01/00010.jpg
# instances: list(len=19)
#   - instances[0] keys: ['bbox_3d', 'bbox_id', 'bbox_label_3d']
#       bbox_3d: list -> 
#       bbox_label_3d: int -> 276
#       bbox_id: int -> 1
# occupancy.npy keys:  (2260, 4)
# scannet/posed_images/scene0191_01/00000.jpg
# (40, 40, 16)