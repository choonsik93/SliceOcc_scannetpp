import os
import json
import glob
import numpy as np
import collections
import open3d as o3d
import cv2
from typing import List, Tuple, Dict

BaseImage = collections.namedtuple("Image", ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"])

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

def qvec2rotmat(qvec):
    return np.array(
        [
            [1 - 2 * qvec[2] ** 2 - 2 * qvec[3] ** 2, 2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3], 2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
            [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3], 1 - 2 * qvec[1] ** 2 - 2 * qvec[3] ** 2, 2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
            [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2], 2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1], 1 - 2 * qvec[1] ** 2 - 2 * qvec[2] ** 2],
        ]
    )

class Image(BaseImage):
    def qvec2rotmat(self):
        return qvec2rotmat(self.qvec)

    def to_transform_mat(self):
        """
        R, t matrix
        """
        R = self.qvec2rotmat()
        t = self.tvec
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t
        return T

    @property
    def world_to_camera(self) -> np.ndarray:
        R = qvec2rotmat(self.qvec)
        t = self.tvec
        world2cam = np.eye(4)
        world2cam[:3, :3] = R
        world2cam[:3, 3] = t
        return world2cam

def read_images_text(path):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadImagesText(const std::string& path)
        void Reconstruction::WriteImagesText(const std::string& path)
    """
    images = {}
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                image_id = int(elems[0])
                qvec = np.array(tuple(map(float, elems[1:5])))
                tvec = np.array(tuple(map(float, elems[5:8])))
                camera_id = int(elems[8])
                image_name = elems[9]
                elems = fid.readline().split()
                xys = np.column_stack([tuple(map(float, elems[0::3])), tuple(map(float, elems[1::3]))])
                point3D_ids = np.array(tuple(map(int, elems[2::3])))
                images[image_id] = Image(id=image_id, qvec=qvec, tvec=tvec, camera_id=camera_id, name=image_name, xys=xys, point3D_ids=point3D_ids)
    return images

def read_colmap_data(data_root_path, scene_id):

    split_path = os.path.join(data_root_path, "data", scene_id, "dslr", "train_test_lists.json")
    with open(split_path, "r") as f:
        train_image_list = json.load(f)["train"]

    cam_rgb_path = glob.glob(os.path.join(data_root_path, "data", scene_id, "dslr", "undistorted_images", "*.JPG"))
    cam_rgb_path.sort()

    cam_infos = json.load(open(os.path.join(data_root_path, "data", scene_id, "dslr", "nerfstudio", "transforms_undistorted.json")))
    fl_x = cam_infos["fl_x"]
    fl_y = cam_infos["fl_y"]
    cx = cam_infos["cx"]
    cy = cam_infos["cy"]
    rgb_W = cam_infos["w"]
    rgb_H = cam_infos["h"]
    cam_intrinsic = np.array([[fl_x, 0, cx], [0, fl_y, cy], [0, 0, 1]])

    colmap_dir = os.path.join(data_root_path, "data", scene_id, "dslr", "colmap", "images.txt")
    images = read_images_text(colmap_dir)

    num_camera = 1
    assert len(cam_rgb_path) % num_camera == 0, "The number of camera poses is not a multiple of %d" % (num_camera)
    cam_ids = ["CAM_FRONT"]
    num_frames = len(cam_rgb_path) // num_camera

    # load all cam data
    CAM_path_pose_timestamp_dict = {}
    for cam_idx, CAM in enumerate(cam_ids):
        CAM_dict = {}
        CAM_timestamps = []
        CAM_poses = []
        CAM_image_paths = []
        CAM_depth_paths = []
        CAM_semantic_paths = []
        CAM_exist_objects = []
        CAM_path_id = []
        path_id = 0
        for i in range(0, num_frames): # 0th frame is not used
            CAM_image_path = cam_rgb_path[i * num_camera + cam_idx]
            CAM_depth_path = CAM_image_path.replace("data", "render_orgpose").replace("undistorted_images", "render_depth").replace("JPG", "png")
            CAM_semantic_path = CAM_image_path.replace("data", "semantics_2d/viz_obj_ids").replace("dslr/undistorted_images", "") + ".png"
            assert os.path.exists(CAM_image_path), "No image file %s" % (CAM_image_path)
            assert os.path.exists(CAM_depth_path), "No depth file %s" % (CAM_depth_path)
            assert os.path.exists(CAM_semantic_path), "No semantic file %s" % (CAM_semantic_path)

            if CAM_image_path.split("/")[-1] not in train_image_list:
                continue

            CAM_objects = cv2.imread(CAM_semantic_path, cv2.IMREAD_UNCHANGED)
            CAM_exist_object = np.unique(CAM_objects)

            CAM_timestamps.append(i)

            extrinsic = None
            filename = os.path.basename(CAM_image_path)

            for key, image in images.items():
                if image.name == filename:
                    extrinsic = image.world_to_camera
                    break

            if extrinsic is None:
                assert False, "No extrinsic matrix found for %s" % (CAM_image_path)

            # cam_pose = cam_poses[i * num_camera + cam_idx]
            # assert cam_pose["file_path"] in CAM_image_path, "Different file path %s %s" % (cam_pose["file_path"], CAM_image_path)
            # extrinsic = np.array(cam_pose["transform_matrix"])

            CAM_poses.append(np.linalg.inv(extrinsic))
            CAM_image_paths.append(CAM_image_path)
            CAM_depth_paths.append(CAM_depth_path)
            CAM_semantic_paths.append(CAM_semantic_path)
            CAM_exist_objects.append(CAM_exist_object)

            # Split cam poses as sequential paths
            if i != 0:
                curr_CAM_pose = CAM_poses[-1]
                prev_CAM_pose = CAM_poses[-2]
                # compute distance between two transforms
                dist = np.linalg.norm(curr_CAM_pose[:3, 3] - prev_CAM_pose[:3, 3])
                # compute angle between two transforms
                angle = np.arccos((np.trace(np.dot(curr_CAM_pose[:3, :3], prev_CAM_pose[:3, :3].T)) - 1) / 2)
                if dist > 0.4 or angle > np.pi / 4.0:
                    path_id += 1
            CAM_path_id.append(path_id)

        CAM_dict["poses"] = CAM_poses
        CAM_dict["timestamps"] = CAM_timestamps
        CAM_dict["image_path"] = CAM_image_paths
        CAM_dict["depth_path"] = CAM_depth_paths
        CAM_dict["semantic_path"] = CAM_semantic_paths
        CAM_dict["exist_object"] = CAM_exist_objects
        CAM_dict["path_id"] = CAM_path_id
        CAM_path_pose_timestamp_dict[CAM] = CAM_dict

    return rgb_W, rgb_H, cam_intrinsic, CAM_path_pose_timestamp_dict

OTHERS = 11
category_to_category_id_dict = {
    "ceiling": 1,
    "floor": 2,
    "wall": 3,
    "window": 4,
    "chair": 5,
    "bed": 6,
    "sofa": 7,
    "table": 8,
    "tv": 9,
    "furniture": 10,
    "objects": OTHERS,
}

def scannet_semantic_to_uniscenes(scannet_semantic_dict, semantic_to_category_dict):
    matching_array = np.zeros(len(scannet_semantic_dict.keys()), dtype=np.uint8)
    for scannet_id, scannet_category in scannet_semantic_dict.items():
        category = semantic_to_category_dict[scannet_category]
        category_id = category_to_category_id_dict[category]
        matching_array[scannet_id] = category_id
    return matching_array

def get_scannet_to_uniscenes(data_root_path):
    # Load Scannet++ pointcloud
    categorized_semantic = os.path.join(data_root_path, "metadata/scannetpp_to_nyunew.csv")
    semantic_to_category_dict = {}
    with open(categorized_semantic, "r") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip().split(",")
            semantic_to_category_dict[line[1]] = line[0]

    scannet_semantic_path = os.path.join(data_root_path, "metadata/semantic_classes.txt")
    scannet_semantic_dict = {}
    with open(scannet_semantic_path, "r") as f:
        id = 0
        lines = f.readlines()
        for line in lines:
            scannet_semantic_dict[id] = line.strip()
            id += 1
    scannet_matching_array = scannet_semantic_to_uniscenes(scannet_semantic_dict, semantic_to_category_dict)

    return scannet_matching_array
    
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

def get_scannetpp_cam_info(data_root, scene_id):
    cam_info_path = os.path.join(data_root, "data", scene_id, "dslr", "nerfstudio", "transforms_undistorted.json")
    with open(cam_info_path, "r") as f:
        cam_info = json.load(f)
    fx, fy, cx, cy = cam_info["fl_x"], cam_info["fl_y"], cam_info["cx"], cam_info["cy"]
    cam_intrinsic = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    height, width = cam_info["h"], cam_info["w"]

    train_frames = cam_info["frames"]
    test_frames = cam_info["test_frames"]

    train_file_names, train_camera_to_worlds = parse_frames(train_frames)
    test_file_names, test_camera_to_worlds = parse_frames(test_frames)

    return cam_intrinsic, height, width, train_file_names, train_camera_to_worlds, test_file_names, test_camera_to_worlds