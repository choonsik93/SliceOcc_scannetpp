import pickle, numpy as np, json
from pathlib import Path
import os
import open3d as o3d
import cv2

OCC_DIR = "/media/sequor/PortableSSD/scannetpp/occupancy_2x"
DATA_DIR = "/media/sequor/PortableSSD/scannetpp"
PKL_FILENAME = "scannetpp_infos_2x_train.pkl"

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

pointcloud_range = [-12.8, -12.8, -0.78, 12.8, 12.8, 3.22]
voxel_dims = [512, 512, 80]
voxel_size = 0.05

def load_pkl(p):
    with open(p, "rb") as f:
        try:
            return pickle.load(f)
        except Exception:
            f.seek(0)
            return pickle.load(f, encoding="latin1")

def np_preview(a, k=3):
    return {
        "type": str(type(a)),
        "dtype": str(a.dtype),
        "shape": list(a.shape),
        "flat_head": a, #a.reshape(-1)[:k].tolist() if a.size else [],
    }

def print_sample(s, idx):
    print(f"\n========== data_list[{idx}] ==========")
    print("keys:", sorted(list(s.keys())))

    # 간단 메타
    for k in ["sample_idx"]:
        if k in s:
            print(f"{k}: {s[k]}")

    # cam matrices
    for k in ["cam2img", "depth_cam2img", "axis_align_matrix"]:
        if k in s and isinstance(s[k], np.ndarray):
            print(f"{k}:", np_preview(s[k]))

    # 이미지 목록
    images = s.get("images", [])
    print(f"images: list(len={len(images)})")
    for j, im in enumerate(images[:2]):  # 앞 2개만
        print(f"  - images[{j}] keys:", sorted(list(im.keys())))
        # 자주 나오는 필드만 프리뷰
        for kk in ["img_path", "extrinsics", "intrinsics", "pose"]:
            if kk in im:
                if isinstance(im[kk], np.ndarray):
                    print(f"      {kk}: {np_preview(im[kk])}")
                else:
                    v = im[kk]
                    print(f"      {kk}: {v if isinstance(v, str) else type(v)}")

    # 인스턴스(박스/라벨)
    instances = s.get("instances", [])
    print(f"instances: list(len={len(instances)})")
    if instances:
        inst0 = instances[0]
        print("  - instances[0] keys:", sorted(list(inst0.keys())))
        for kk, vv in inst0.items():
            if isinstance(vv, np.ndarray):
                print(f"      {kk}: {np_preview(vv)}")
            else:
                print(f"      {kk}: {type(vv).__name__} -> {vv if isinstance(vv, (int,str,float)) else ''}")

    # occupancy
    cam2img = s.get("cam2img")
    global2occ = s.get("axis_align_matrix")

    occ_path = os.path.join(OCC_DIR, s["sample_idx"], "occupancy.npy")
    occ_data = np.load(occ_path)
    o3d_pcd = o3d.geometry.PointCloud()
    occ_points = occ_data[:, :3].astype(np.float32)
    occ_points = (occ_points + 0.5) * voxel_size + np.array(pointcloud_range[:3])[None, :]
    # o3d_pcd.points = o3d.utility.Vector3dVector(occ_data[:, :3].astype(np.float32))
    o3d_pcd.points = o3d.utility.Vector3dVector(occ_points)
    occ_labels = occ_data[:, 3]
    occ_colors = np.zeros((occ_labels.shape[0], 3), dtype=np.float32)
    for class_id in np.unique(occ_labels):
        occ_colors[occ_labels == class_id] = np.random.rand(3)
    o3d_pcd.colors = o3d.utility.Vector3dVector(occ_colors)
    o3d_voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(o3d_pcd, voxel_size=voxel_size)
    #o3d.visualization.draw_geometries([o3d_voxel_grid])

    def draw_camera_with_image(
        image_path,
        K, R, t, w, h,
        scale=1,
        color=[0.8, 0.2, 0.8],
        draw_axis=True,
        draw_frustum=True,
    ):
        """Create axis, plane and pyramed geometries in Open3D format.
        Args:
            image_path: path to the image to be textured on the plane
            K: camera intrinsics
            R: rotation matrix (camera_to_world)
            t: translation vector (camera_to_world)
            w: image width
            h: image height
            scale: camera model scale
            color: color of the camera
            draw_axis: whether to draw axis
            draw_frustum: whether to draw frustum
        Returns:
            List of Open3D geometries (axis, plane and pyramid)
        """
        # load image
        Kinv = np.linalg.inv(K)
        img = cv2.imread(image_path)
        h, w = img.shape[:2]

        # 4x4 T (camera_to_world)
        T = np.column_stack((R, t.reshape(3,)))
        T = np.vstack((T, (0, 0, 0, 1)))

        geoms = []

        if draw_frustum:
            pix = np.array([
                [0, 0, 0],
                [0, 0, 1],
                [w, 0, 1],
                [0, h, 1],
                [w, h, 1],
            ], dtype=float).T  # 3x5
            rays = Kinv @ pix[:, 1:]                 # 3x4 (z=1)
            corners_cam = (rays * scale).T           # 4x3
            points_world = np.vstack([np.zeros((1,3)), corners_cam])  # [C, TL, TR, BR, BL] in cam
            points_world = (R @ points_world.T + t.reshape(3,1)).T

            lines = np.array([[0,1],[0,2],[0,3],[0,4],[1,2],[2,4],[4,3],[3,1]], dtype=np.int32)
            colors = np.tile(np.array(color, float), (lines.shape[0], 1))
            line_set = o3d.geometry.LineSet(
                points=o3d.utility.Vector3dVector(points_world),
                lines=o3d.utility.Vector2iVector(lines))
            line_set.colors = o3d.utility.Vector3dVector(colors)
            geoms.append(line_set)

        if draw_axis:
            axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.8 * scale)
            axis.transform(T)
            geoms.append(axis)

        pix_corners = np.array([[0,0,1],[w,0,1],[w,h,1],[0,h,1]], dtype=float).T  # 3x4
        dirs = Kinv @ pix_corners     # 3x4, z=1
        plane_cam = (dirs * scale).T  # 4x3

        plane_world = (R @ plane_cam.T + t.reshape(3,1)).T  # 4x3
        verts = plane_world
        # tris  = np.array([[0,1,2],[0,2,3]], dtype=np.int32)
        # tri_uvs = np.array([
        #     [0.0, 0.0], [1.0, 0.0], [1.0, 1.0],   # 첫 삼각형
        #     [0.0, 0.0], [1.0, 1.0], [0.0, 1.0],   # 둘째 삼각형
        # ], dtype=float)
        tris  = np.array([[0,2,1],[0,3,2]], dtype=np.int32)
        tri_uvs = np.array([
            [0.0, 0.0], [1.0, 1.0], [1.0, 0.0],   # 삼각형 1: (0,2,1)
            [0.0, 0.0], [0.0, 1.0], [1.0, 1.0],   # 삼각형 2: (0,3,2)
        ], dtype=float)

        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(verts)
        mesh.triangles = o3d.utility.Vector3iVector(tris)
        mesh.triangle_uvs = o3d.utility.Vector2dVector(tri_uvs)
        mesh.triangle_material_ids = o3d.utility.IntVector([0, 0])
        mesh.compute_vertex_normals()

        if img.dtype != np.uint8:
            tex = (np.clip(img, 0, 1) * 255).astype(np.uint8)
        else:
            tex = img
        mesh.textures = [o3d.geometry.Image(tex)]
        geoms.append(mesh)

        return geoms

    o3d_cam_axis_list = []
    for j, im in enumerate(images):
        img_path = im["img_path"]
        depth_path = im["depth_path"]
        cam2global = im["cam2global"]
        cam2occ = global2occ @ cam2global
        # o3d_cam_axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[0,0,0])
        # o3d_cam_axis.transform(cam2occ)
        # o3d_cam_axis_list.append(o3d_cam_axis)

        o3d_geoms = draw_camera_with_image(
            os.path.join(DATA_DIR, img_path),
            cam2img[:3, :3], cam2occ[:3, :3], cam2occ[:3, 3], 1752, 1168,
            scale=0.1,
            color=[0.8, 0.2, 0.8],
            draw_axis=True,
            draw_frustum=True,
        )

        o3d_cam_axis_list.extend(o3d_geoms)

    o3d_global_axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=3.2, origin=[0,0,0])
    
    o3d.visualization.draw_geometries(o3d_cam_axis_list + [o3d_voxel_grid, o3d_global_axis])

if __name__ == "__main__":
    data = load_pkl(os.path.join(DATA_DIR, PKL_FILENAME))
    assert isinstance(data, dict) and "data_list" in data
    print(data["metainfo"])
    dl = data["data_list"]

    # 보고 싶은 인덱스 지정
    inspect_idxs = [0, 1, 2]      # <- 여기 바꿔서 사용
    # 혹은 sample_idx로 찾고 싶으면 아래 사용:
    # target_sample_idx = "scene0000_00"
    # inspect_idxs = [i for i,s in enumerate(dl) if s.get("sample_idx")==target_sample_idx][:1]

    for i in inspect_idxs:
        print_sample(dl[i], i)

# python -m embodiedscan.converter.sample_scannetpp_pkl_file