import os
import json
import torch
import numpy as np
from tqdm import tqdm
from .scannetpp_io import get_scannet_to_uniscenes, get_scannetpp_cam_info, NYU_COLORS
from .scannetpp_visible_mask import build_visible_mask_numba
import open3d as o3d
import pickle

# Label IDs
NYU_FLOOR_ID = 2
NYU_WALL_ID  = 3
NYU_CEIL_ID  = 1
IGNORE_ID    = 255
PONINTCLOUD_RANGE = [-6.0, -6.0, -0.78, 6.0, 6.0, 3.22]
VOXEL_DIMS = [240, 240, 80]
VOXEL_SIZE = 0.05


def _rotation_from_a_to_b(a, b):
    """Return 3x3 rotation matrix that rotates vector a to vector b."""
    a = a / (np.linalg.norm(a) + 1e-9)
    b = b / (np.linalg.norm(b) + 1e-9)
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    if np.linalg.norm(v) < 1e-12:
        # parallel or anti-parallel
        if c > 0.0:
            return np.eye(3, dtype=np.float64)
        else:
            # 180 deg around any axis orthogonal to a
            # find an orthogonal vector
            axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
            if abs(a[0]) > 0.9:
                axis = np.array([0.0, 1.0, 0.0], dtype=np.float64)
            v = np.cross(a, axis)
            v = v / (np.linalg.norm(v) + 1e-9)
            K = np.array([[0, -v[2], v[1]],
                          [v[2], 0, -v[0]],
                          [-v[1], v[0], 0]], dtype=np.float64)
            return -np.eye(3) + 2 * np.outer(v, v)  # Rodrigues at 180°
    K = np.array([[0, -v[2], v[1]],
                  [v[2], 0, -v[0]],
                  [-v[1], v[0], 0]], dtype=np.float64)
    R = np.eye(3, dtype=np.float64) + K + K @ K * (1.0 / (1.0 + c + 1e-12))
    return R

def _estimate_R_up_from_floor(coords, labels, floor_id=NYU_FLOOR_ID, min_pts=200):
    """Estimate rotation to align floor normal to +Z using PCA."""
    mask = (labels == floor_id)
    if mask.sum() < min_pts:
        return np.eye(3, dtype=np.float64), False
    P = coords[mask]  # (N,3)
    # PCA: smallest eigenvector is normal
    C = np.cov(P.T)
    eigvals, eigvecs = np.linalg.eigh(C)
    n = eigvecs[:, np.argmin(eigvals)]
    # ensure upward (+Z) direction
    if n[2] < 0:
        n = -n
    R_up = _rotation_from_a_to_b(n, np.array([0., 0., 1.], dtype=np.float64))
    return R_up, True

def _estimate_R_yaw_from_walls(coords_uplifted, labels, wall_id=NYU_WALL_ID, min_pts=300):
    """Estimate yaw so that walls align with world X/Y axes (snap to 90°)."""
    mask = (labels == wall_id)
    if mask.sum() < min_pts:
        return np.eye(3, dtype=np.float64), False
    Q = coords_uplifted[mask]  # after R_up
    XY = Q[:, :2]
    # PCA on XY
    C = np.cov(XY.T)
    eigvals, eigvecs = np.linalg.eigh(C)
    # principal axis (largest eigenvalue)
    v = eigvecs[:, np.argmax(eigvals)]
    # angle of principal axis relative to +X
    theta = np.arctan2(v[1], v[0])
    # snap to nearest 90 deg
    snapped = np.round(theta / (np.pi/2.0)) * (np.pi/2.0)
    yaw = theta - snapped  # rotate by -yaw to align
    c, s = np.cos(-yaw), np.sin(-yaw)
    Rz = np.array([[c, -s, 0.],
                   [s,  c, 0.],
                   [0., 0., 1.]], dtype=np.float64)
    return Rz, True

def multi_plane_segment_normals_XY(coords_upright_wall, distance_thresh=0.02,
                                   ransac_n=3, num_iterations=2000,
                                   min_inliers=1000, max_planes=10,
                                   merge_normal_eps_deg=10.0):
    """
    Run iterative plane segmentation on wall points and return unique horizontal normals.
    coords_upright_wall: (N,3) wall-only points after R_up
    Returns: list of 2D unit vectors on XY from plane normals (unique/merged)
    """
    if coords_upright_wall.shape[0] < min_inliers:
        return []

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(coords_upright_wall)

    normals_xy = []
    rest = pcd

    for _ in range(max_planes):
        if np.asarray(rest.points).shape[0] < min_inliers:
            break
        plane_model, inliers = rest.segment_plane(distance_threshold=distance_thresh,
                                                  ransac_n=ransac_n,
                                                  num_iterations=num_iterations)
        if len(inliers) < min_inliers:
            break
        a, b, c, d = plane_model  # ax+by+cz+d=0
        n = np.array([a, b, c], dtype=np.float64)
        # ignore near-vertical? (we want wall normals ≈ horizontal)
        # actually wall normals are horizontal → z-component small
        if abs(n[2]) < 0.9:  # keep horizontal-ish normals
            t = n[:2]
            norm_xy = np.linalg.norm(t) + 1e-9
            t = t / norm_xy
            # canonicalize direction (θ and θ+π same)
            if t[0] < 0:
                t = -t
            normals_xy.append(t)

        # remove inliers and continue
        inlier_cloud = rest.select_by_index(inliers)
        rest = rest.select_by_index(inliers, invert=True)

    # merge similar directions
    merged = []
    def angle_deg(u, v):
        # angle on the projective circle (θ ~ θ+π)
        dot = float(np.clip(np.dot(u, v), -1.0, 1.0))
        ang = np.degrees(np.arccos(dot))
        return min(ang, 180.0 - ang)

    for t in normals_xy:
        placed = False
        for i, g in enumerate(merged):
            if angle_deg(t, g) < merge_normal_eps_deg:
                merged[i] = (g + t) / (np.linalg.norm(g + t) + 1e-9)
                placed = True
                break
        if not placed:
            merged.append(t)

    return merged

def estimate_R_yaw_from_planes(coords_upright, labels, wall_id=3):
    """
    Estimate yaw (Rz) from multiple wall planes.
    coords_upright: (N,3) after R_up
    labels: (N,)
    """
    wall_pts = coords_upright[labels == wall_id]
    dirs = multi_plane_segment_normals_XY(wall_pts)
    if len(dirs) == 0:
        return np.eye(3), False

    # choose the strongest direction as x-axis target
    # (you could also histogram angles and pick peaks)
    main = dirs[0]
    # yaw to align 'main' with +X
    theta = np.arctan2(main[1], main[0])
    snapped = np.round(theta / (np.pi/2.0)) * (np.pi/2.0)
    yaw = theta - snapped
    c, s = np.cos(-yaw), np.sin(-yaw)
    Rz = np.array([[c, -s, 0.0],
                   [s,  c, 0.0],
                   [0.0, 0.0, 1.0]], dtype=np.float64)
    return Rz, True

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

scannetpp_to_nyu = get_scannet_to_uniscenes(data_root_dir)

def compute_scene_extent_with_plot(data_root_dir, pointcloud_dir, scannetpp_to_nyu, scene_id_list, split="train",
                                   pointcloud_range=[-12.8, -12.8, -0.78, 12.8, 12.8, 3.22],
                                   voxel_dims=[512, 512, 80], voxel_size=0.05):
    
    import matplotlib.pyplot as plt
    
    x_min, y_min, z_min, x_max, y_max, z_max = pointcloud_range
    nx, ny, nz = voxel_dims
    count = 0
    all_extents = []

    for scene_id in tqdm(scene_id_list, desc=f"Computing extents ({split})"):
        output_dir = os.path.join(data_root_dir, "occupancy_2x", scene_id)
        os.makedirs(output_dir, exist_ok=True)
        # load point cloud 
        pointcloud_data = torch.load(os.path.join(pointcloud_dir, scene_id + ".pth"), weights_only=False)
        coords = pointcloud_data["sampled_coords"].astype(np.float32)
        scene_extent = np.max(coords, axis=0) - np.min(coords, axis=0)
        all_extents.append(scene_extent)

        if scene_extent[0] > (x_max - x_min) or scene_extent[1] > (y_max - y_min) or scene_extent[2] > (z_max - z_min):
            count += 1

    all_extents = np.array(all_extents)
    print(f"Total {count} scenes exceed the defined range.")
    print(f"Mean extent: {np.mean(all_extents, axis=0)}")
    print(f"Max extent:  {np.max(all_extents, axis=0)}")
    print(f"Min extent:  {np.min(all_extents, axis=0)}")

    # --- 그래프 1: 각 축별 히스토그램 ---
    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    axs[0].hist(all_extents[:, 0], bins=30, color='r', alpha=0.7)
    axs[0].set_title("X extent")
    axs[1].hist(all_extents[:, 1], bins=30, color='g', alpha=0.7)
    axs[1].set_title("Y extent")
    axs[2].hist(all_extents[:, 2], bins=30, color='b', alpha=0.7)
    axs[2].set_title("Z extent")
    plt.suptitle(f"Scene extent distribution ({split})")
    plt.tight_layout()
    plt.show()

    # --- 그래프 2: 3D scatter ---
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(all_extents[:, 0], all_extents[:, 1], all_extents[:, 2], alpha=0.7, c='orange')
    ax.set_xlabel("Extent X (m)")
    ax.set_ylabel("Extent Y (m)")
    ax.set_zlabel("Extent Z (m)")
    ax.set_title(f"Scene extent scatter ({split})")
    plt.show()

    # --- 저장 ---
    save_path = os.path.join(f"scene_extent_{split}.npy")
    np.save(save_path, all_extents)
    print(f"Saved extent data to {save_path}")

def compute_scene_extent(data_root_dir, pointcloud_dir, scannetpp_to_nyu, scene_id_list, split="train",
                                 pointcloud_range=[-12.8, -12.8, -0.78, 12.8, 12.8, 3.22], voxel_dims=[512, 512, 80], voxel_size=0.05):
    
    x_min, y_min, z_min, x_max, y_max, z_max = pointcloud_range
    nx, ny, nz = voxel_dims
    count = 0
    for scene_id in tqdm(scene_id_list):

        output_dir = os.path.join(data_root_dir, "occupancy_2x", scene_id)
        os.makedirs(output_dir, exist_ok=True)
        # load point cloud 
        pointcloud_data = torch.load(os.path.join(pointcloud_dir, scene_id + ".pth"), weights_only=False)
        coords = pointcloud_data["sampled_coords"].astype(np.float32)
        scene_extent = np.max(coords, axis=0) - np.min(coords, axis=0)
        if scene_extent[0] > (x_max - x_min) or scene_extent[1] > (y_max - y_min) or scene_extent[2] > (z_max - z_min):
            count += 1
    print(f"Total {count} scenes exceed the defined range.")

def generate_scannetpp_occupancy(data_root_dir, pointcloud_dir, scannetpp_to_nyu, scene_id_list, split="train",
                                 pointcloud_range=PONINTCLOUD_RANGE, voxel_dims=VOXEL_DIMS, voxel_size=VOXEL_SIZE, debug=False):
    
    x_min, y_min, z_min, x_max, y_max, z_max = pointcloud_range
    nx, ny, nz = voxel_dims

    for scene_id in tqdm(scene_id_list):

        cam_intrinsic, height, width, train_file_names, train_camera_to_worlds, test_file_names, test_camera_to_worlds = get_scannetpp_cam_info(data_root_dir, scene_id)

        output_dir = os.path.join(data_root_dir, "occupancy_2x", scene_id)
        os.makedirs(output_dir, exist_ok=True)

        # if os.path.exists(os.path.join(output_dir, "occupancy.npy")) and \
        #       os.path.exists(os.path.join(output_dir, "axis_align_matrix.npy")) and \
        #         os.path.exists(os.path.join(output_dir, "visible_occupancy.pkl")):
        #     continue

        # load point cloud 
        pointcloud_data = torch.load(os.path.join(pointcloud_dir, scene_id + ".pth"), weights_only=False)
        coords = pointcloud_data["sampled_coords"].astype(np.float32)
        labels = pointcloud_data["sampled_labels"].astype(np.int32)
        org_labels = labels.copy()
        nyu_labels = scannetpp_to_nyu[labels]
        nyu_labels[org_labels == -100] = 255  # mark unknown
        objects = pointcloud_data["sampled_instance_anno_id"].astype(np.int32)
        scene_center = np.median(coords, axis=0)
        scene_extent = np.max(coords, axis=0) - np.min(coords, axis=0)
        #print("Scene center:", scene_center, "extent:", scene_extent, np.min(coords, axis=0), np.max(coords, axis=0))

        # 1) estimate R_up from floor
        R_up, ok_up = _estimate_R_up_from_floor(coords, nyu_labels, floor_id=NYU_FLOOR_ID)

        # lift by R_up
        coords_up = (R_up @ coords.T).T

        # 2) estimate yaw from walls
        # R_yaw, ok_yaw = _estimate_R_yaw_from_walls(coords_up, nyu_labels, wall_id=NYU_WALL_ID)
        R_yaw, ok_yaw = estimate_R_yaw_from_planes(coords_up, nyu_labels, wall_id=NYU_WALL_ID)

        A = np.eye(4, dtype=np.float64)
        A[:3, :3] = R_yaw @ R_up
        axis_align_matrix = A
        coords_h = np.concatenate([coords.astype(np.float32), np.ones((coords.shape[0], 1), dtype=np.float32)], axis=1)
        coords = (axis_align_matrix @ coords_h.T).T[:, :3].astype(np.float32)

        scene_offset = (np.max(coords, axis=0) + np.min(coords, axis=0)) / 2.0
        scene_offset[2] = np.min(coords[:, 2]) + 0.5  # raise floor to near z_min
        axis_align_matrix[:3, 3] = -scene_offset

        coords = pointcloud_data["sampled_coords"].astype(np.float32)
        coords_h = np.concatenate([coords.astype(np.float32), np.ones((coords.shape[0], 1), dtype=np.float32)], axis=1)
        coords = (axis_align_matrix @ coords_h.T).T[:, :3].astype(np.float32)
        # print(np.min(coords, axis=0), np.max(coords, axis=0), np.median(coords, axis=0))

        if debug:
            o3d_pcd = o3d.geometry.PointCloud()
            o3d_pcd.points = o3d.utility.Vector3dVector(coords.astype(np.float32))
            occ_labels = nyu_labels
            occ_labels[occ_labels == 255] = 12
            colors = NYU_COLORS[occ_labels][:, :3] / 255.0
            o3d_pcd.colors = o3d.utility.Vector3dVector(colors)
            #o3d_voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(o3d_pcd, voxel_size=1)
            o3d_cam_axis_list = []
            for idx in range(len(train_camera_to_worlds)):
                o3d_axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[0,0,0])
                cam2global = train_camera_to_worlds[idx]
                cam2occ = axis_align_matrix @ cam2global
                o3d_axis.transform(cam2occ)
                o3d_cam_axis_list.append(o3d_axis)
            o3d_global_axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=6.0, origin=[0,0,0])
            o3d.visualization.draw_geometries([o3d_pcd, o3d_global_axis] + o3d_cam_axis_list)

        in_range = (
            (coords[:, 0] >= x_min) & (coords[:, 0] < x_max) &
            (coords[:, 1] >= y_min) & (coords[:, 1] < y_max) &
            (coords[:, 2] >= z_min) & (coords[:, 2] < z_max)
        )
        coords = coords[in_range]
        nyu_labels = nyu_labels[in_range]
        
        # map to voxel indices
        # idx = floor((x - min) / voxel_size)
        gx = np.floor((coords[:, 0] - x_min) / voxel_size).astype(np.int32)
        gy = np.floor((coords[:, 1] - y_min) / voxel_size).astype(np.int32)
        gz = np.floor((coords[:, 2] - z_min) / voxel_size).astype(np.int32)

        # # clamp for numerical safety (should already be in-range)
        gx = np.clip(gx, 0, nx - 1)
        gy = np.clip(gy, 0, ny - 1)
        gz = np.clip(gz, 0, nz - 1)
        
        # flatten 3D index -> 1D key for grouping
        lin = np.ravel_multi_index((gx, gy, gz), dims=(nx, ny, nz))

        # group by voxel (unique lin)
        uniq_lin, inv = np.unique(lin, return_inverse=True)

        # majority vote per voxel
        # fast bincount on each group
        # we’ll build an array of selected (voted) labels per unique voxel
        voted_labels = np.empty(uniq_lin.shape[0], dtype=np.int32)
        for i in range(uniq_lin.shape[0]):
            mask = (inv == i)
            counts = np.bincount(nyu_labels[mask])
            voted_labels[i] = np.argmax(counts)
            # # ignore negative / 255 labels if you have them (optional)
            # if np.any(labels[mask] >= 0):
            #     counts = np.bincount(labels[mask][labels[mask] >= 0])
            #     voted_labels[i] = np.argmax(counts)
            # else:
            #     # if all are invalid -> mark unknown (255)
            #     voted_labels[i] = 255

        # recover (gx,gy,gz) for each unique lin
        ux, uy, uz = np.unravel_index(uniq_lin, (nx, ny, nz))

        # pack as [ux, uy, uz, cls]
        occ = np.stack([ux, uy, uz, voted_labels], axis=1).astype(np.int32)

        if debug:
            o3d_axis_list = []
            for idx in range(len(train_camera_to_worlds)):
                cam2global = train_camera_to_worlds[idx]
                cam2occ = axis_align_matrix @ cam2global
                o3d_axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[0,0,0])
                o3d_axis.transform(cam2occ)
                o3d_axis_list.append(o3d_axis)
            o3d_global_pcd = o3d.geometry.PointCloud()
            o3d_global_pcd.points = o3d.utility.Vector3dVector((occ[:, :3].astype(np.float32) + 0.5) * voxel_size + np.array(pointcloud_range[:3])[None, :])
            o3d_global_voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(o3d_global_pcd, voxel_size=voxel_size)
            o3d_global_axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=6.0, origin=[0,0,0])
            o3d.visualization.draw_geometries([o3d_global_voxel_grid, o3d_global_axis] + o3d_axis_list)

        # 4x downsampled occupancy for visible mask computation
        occ_dense = np.zeros((nx // 4, ny // 4, nz // 4), dtype=np.bool_)
        occ_dense[occ[:, 0] // 4, occ[:,1] // 4, occ[:,2] // 4] = True

        visible_occupancy_list = []

        origin = np.array([x_min, y_min, z_min], dtype=np.float64)
        voxel_size_3d = np.array([voxel_size * 4, voxel_size * 4, voxel_size * 4], dtype=np.float64)
        for idx in range(len(train_camera_to_worlds)):
            filename = train_file_names[idx]
            cam2global = train_camera_to_worlds[idx]
            cam2occ = axis_align_matrix @ cam2global
            occ2cam = np.linalg.inv(cam2occ)
            visible_mask = build_visible_mask_numba(
                occ_dense, origin, voxel_size_3d, cam_intrinsic, occ2cam[:3, :3], occ2cam[:3, 3], (width, height), stride=1, max_range=-1.0
            )
            visible_occupancy_list.append({
                "img_path": filename,
                "visible_occupancy": visible_mask,
            })
            if debug:
                # Visualize occupancy
                o3d_global_pcd = o3d.geometry.PointCloud()
                o3d_global_pcd.points = o3d.utility.Vector3dVector((occ[:, :3].astype(np.float32) + 0.5) * voxel_size + np.array(pointcloud_range[:3])[None, :])
                o3d_global_voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(o3d_global_pcd, voxel_size=voxel_size)
                ijk = np.argwhere(visible_mask)
                o3d_pcd = o3d.geometry.PointCloud()
                o3d_pcd.points = o3d.utility.Vector3dVector((ijk.astype(np.float32) + 0.5) * voxel_size_3d[0] + np.array(pointcloud_range[:3])[None, :])
                o3d_pcd.paint_uniform_color([1, 0, 0])
                o3d_voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(o3d_pcd, voxel_size=voxel_size_3d[0])

                o3d_axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[0,0,0])
                o3d_axis.transform(cam2occ)

                o3d.visualization.draw_geometries([o3d_voxel_grid, o3d_global_voxel_grid, o3d_axis])

        # o3d_pcd = o3d.geometry.PointCloud()
        # o3d_pcd.points = o3d.utility.Vector3dVector(occ[:, :3].astype(np.float32))
        # occ_labels = occ[:, 3]
        # occ_labels[occ_labels == 255] = 12
        # colors = NYU_COLORS[occ_labels][:, :3] / 255.0
        # o3d_pcd.colors = o3d.utility.Vector3dVector(colors)
        # o3d_voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(o3d_pcd, voxel_size=1)

        # o3d_global_axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=40, origin=[0,0,0])
        # o3d.visualization.draw_geometries([o3d_global_axis, o3d_voxel_grid])

        save_path = os.path.join(output_dir, "occupancy.npy")
        np.save(save_path, occ)
        save_path = os.path.join(output_dir, "axis_align_matrix.npy")
        np.save(save_path, axis_align_matrix)
        save_path = os.path.join(output_dir, "visible_occupancy.pkl")
        with open(save_path, "wb") as f:
            pickle.dump(visible_occupancy_list, f, protocol=pickle.HIGHEST_PROTOCOL)
        # print(f"wrote {occ.shape[0]} voxels -> {save_path}")

#compute_scene_extent_with_plot(data_root_dir, pointcloud_dir, scannetpp_to_nyu, train_scene_ids + val_scene_ids, split="all")
generate_scannetpp_occupancy(data_root_dir, pointcloud_dir, scannetpp_to_nyu, train_scene_ids, split="train")
#compute_scene_extent(data_root_dir, pointcloud_dir, scannetpp_to_nyu, train_scene_ids, split="train")
generate_scannetpp_occupancy(data_root_dir, pointcloud_dir, scannetpp_to_nyu, val_scene_ids, split="val")
#compute_scene_extent(data_root_dir, pointcloud_dir, scannetpp_to_nyu, val_scene_ids, split="val")


