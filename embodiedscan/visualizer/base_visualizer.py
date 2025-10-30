import os

from mmengine.dist import master_only
from mmengine.visualization import Visualizer

from embodiedscan.registry import VISUALIZERS

try:
    import open3d as o3d

    from embodiedscan.visualization.utils import _9dof_to_box, nms_filter
except ImportError:
    o3d = None

import numpy as np

'''
from mayavi import mlab
import mayavi
import argparse, torch, os, json
import shutil
import numpy as np
import mmcv
from embodiedscan.models.losses.occ_loss import occ_multiscale_supervision
from PIL import ImageGrab
'''

def get_grid_coords(dims, resolution):
    """
    :param dims: the dimensions of the grid [x, y, z] (i.e. [256, 256, 32])
    :return coords_grid: is the center coords of voxels in the grid
    """

    g_xx = np.arange(0, dims[0]) # [0, 1, ..., 256]
    # g_xx = g_xx[::-1]
    g_yy = np.arange(0, dims[1]) # [0, 1, ..., 256]
    # g_yy = g_yy[::-1]
    g_zz = np.arange(0, dims[2]) # [0, 1, ..., 32]

    # Obtaining the grid with coords...
    xx, yy, zz = np.meshgrid(g_xx, g_yy, g_zz)
    coords_grid = np.array([xx.flatten(), yy.flatten(), zz.flatten()]).T
    coords_grid = coords_grid.astype(np.float32)
    print(resolution)
    resolution = np.array(resolution, dtype=np.float32).reshape([1, 3])

    coords_grid = (coords_grid * resolution) + resolution / 2

    return coords_grid


@VISUALIZERS.register_module()
class EmbodiedScanBaseVisualizer(Visualizer):
    """EmbodiedScan Base Visualizer. Method to visualize 3D scenes and Euler
    boxes.

    Args:
        name (str): Name of the visualizer. Defaults to 'visualizer'.
        save_dir (str, optional): Directory to save visualizations.
            Defaults to None.
        vis_backends (list[ConfigType], optional):
            List of visualization backends to use. Defaluts to None.
    """

    def __init__(self,
                 name: str = 'visualizer',
                 save_dir: str = None,
                 vis_backends=None) -> None:
        super().__init__(name=name,
                         vis_backends=vis_backends,
                         save_dir=save_dir)

        if o3d is None:
            raise ImportError('Please install open3d.')

    @staticmethod
    def get_root_dir(img_path):
        """Get the root directory of the dataset."""
        if 'posed_images' in img_path:
            return img_path.split('posed_images')[0]
        if 'sequence' in img_path:
            return img_path.split('sequence')[0]
        if 'matterport_color_images' in img_path:
            return img_path.split('matterport_color_images')[0]
        if 'undistorted_images' in img_path:
            return img_path.split('undistorted_images')[0]
        raise ValueError('Custom datasets are not supported.')

    @staticmethod
    def get_ply(root_dir, scene_name):
        """Get the path of the ply file."""
        s = scene_name.split('/')
        if len(s) == 2:
            dataset, region = s
        else:
            dataset, building, region = s
        if dataset == 'scannet':
            filepath = os.path.join(root_dir, 'scans', region,
                                    f'{region}_vh_clean.ply')
        elif dataset == '3rscan':
            filepath = os.path.join(root_dir, 'mesh.refined.v2.obj')
        elif dataset == 'matterport3d':
            filepath = os.path.join(root_dir, 'region_segmentations',
                                    f'{region}.ply')
        else:
            raise NotImplementedError
        return filepath

    @master_only
    def visualize_scene(self,
                        data_samples,
                        class_filter=None,
                        nms_args=dict(iou_thr=0.15,
                                      score_thr=0.075,
                                      topk_per_class=10)):
        """Visualize the 3D scene with 3D boxes.

        Args:
            data_samples (list[:obj:`Det3DDataSample`]):
                The output of the model.
            class_filter (int, optional): Class filter for visualization.
                Default to None to show all classes.
            nms_args (dict): NMS arguments for filtering boxes.
                Defaults to dict(iou_thr = 0.15,
                                 score_thr = 0.075,
                                 topk_per_class = 10).
        """
        assert len(data_samples) == 1
        data_sample = data_samples[0]

        metainfo = data_sample.metainfo
        pred = data_sample.pred_instances_3d
        gt = data_sample.eval_ann_info

        if not hasattr(pred, 'labels_3d'):
            assert gt['gt_labels_3d'].shape[0] == 1
            gt_label = gt['gt_labels_3d'][0].item()
            _ = pred.bboxes_3d.tensor.shape[0]
            pseudo_label = pred.bboxes_3d.tensor.new_ones(_, ) * gt_label
            pred.labels_3d = pseudo_label
        pred_box, pred_label = nms_filter(pred, **nms_args)

        root_dir = self.get_root_dir(metainfo['img_path'][0])
        ply_file = self.get_ply(root_dir, metainfo['scan_id'])
        axis_align_matrix = metainfo['axis_align_matrix']

        mesh = o3d.io.read_triangle_mesh(ply_file, True)
        mesh.transform(axis_align_matrix)
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame()
        
        boxes = []
        # pred 3D box
        n = pred_box.shape[0]
        for i in range(n):
            box = pred_box[i]
            label = pred_label[i]
            if class_filter is not None and label != class_filter:
                continue
            box_geo = _9dof_to_box(box, color=(255, 0, 0))
            boxes.append(box_geo)
        # gt 3D box
        m = gt['gt_bboxes_3d'].tensor.shape[0]
        for i in range(m):
            box = gt['gt_bboxes_3d'].tensor[i]
            label = gt['gt_labels_3d'][i]
            if class_filter is not None and label != class_filter:
                continue
            box_geo = _9dof_to_box(box, color=(0, 255, 0))
            boxes.append(box_geo)

        o3d.visualization.draw_geometries([mesh, frame] + boxes)
    
    @master_only
    def visualize_occupancy(self,
                            data_samples,
                            voxel_size=[0.16, 0.16, 0.16],  # voxel size in the real world
                            ):

        assert len(data_samples) == 1
        data_sample = data_samples[0]

        metainfo = data_sample.metainfo
        
        #root_dir = self.get_root_dir(metainfo['img_path'][0])
        # ply_file = self.get_ply(root_dir, metainfo['scan_id'])
        # axis_align_matrix = metainfo['axis_align_matrix']

        # Compute the voxels coordinates
        voxels = data_sample.pred_occupancy.cpu().detach()

        COLORS = np.array([
            [ 22, 191, 206, 255], # 00 free
            [214,  38,  40, 255], # 01 ceiling
            [ 43, 160,  43, 255], # 02 floor
            [158, 216, 229, 255], # 03 wall
            [114, 158, 206, 255], # 04 window
            [204, 204,  91, 255], # 05 chair
            [255, 186, 119, 255], # 06 bed
            [147, 102, 188, 255], # 07 sofa
            [ 30, 119, 181, 255], # 08 table
            [188, 188,  33, 255], # 09 tvs
            [255, 127,  12, 255], # 10 furniture
            [196, 175, 214, 255], # 11 objects
            [153, 153, 153, 255], # 12 unknown
        ]).astype(np.uint8)

        nonzero_indices = np.argwhere(voxels.numpy() != 0)
        nonzero_labels = voxels.numpy()[nonzero_indices[:, 0], nonzero_indices[:, 1], nonzero_indices[:, 2]]
        occ_colors = COLORS[nonzero_labels.astype(np.int32), :3].astype(np.float32) / 255.0
        o3d_pcd = o3d.geometry.PointCloud()
        o3d_pcd.points = o3d.utility.Vector3dVector(nonzero_indices.astype(np.float32))
        o3d_pcd.colors = o3d.utility.Vector3dVector(occ_colors)
        o3d_voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(o3d_pcd, voxel_size=1)
        #o3d_axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=40, origin=[0,0,0])
        #o3d.visualization.draw_geometries([o3d_voxel_grid, o3d_axis])

        def save_voxelgrid_png(o3d_voxel_grid,
                            out_path,
                            width=1600,
                            height=1200,
                            bg_color=(0, 0, 0),
                            front=(0.5, -0.5, -1.0),
                            lookat=None,
                            up=(0, -1, 0),
                            zoom=0.7):
            # 비가시 모드 창 생성
            vis = o3d.visualization.Visualizer()
            vis.create_window(visible=False, width=width, height=height)
            vis.add_geometry(o3d_voxel_grid)

            # 렌더 옵션
            opt = vis.get_render_option()
            opt.background_color = np.asarray(bg_color, dtype=np.float64)

            eye    = np.array([20, 20, -60], dtype=np.float64)
            lookat = np.array([20, 20, 0], dtype=np.float64)  # 정면(아래)으로 볼 지점
            front  = (lookat - eye); front = front / np.linalg.norm(front)  # (0,0,-1)로 됨
            up     = np.array([0.0, 1.0, 0.0], dtype=np.float64)  # 필요시 [0,-1,0]로 바꿔보면 느낌 달라짐

            # 카메라 뷰 세팅 (대충 중앙 바라보게)
            ctr = vis.get_view_control()
            if lookat is None:
                aabb = o3d_voxel_grid.get_axis_aligned_bounding_box()
                lookat = aabb.get_center()
            ctr.set_front(front)
            ctr.set_up(up)
            ctr.set_lookat(lookat)
            ctr.set_zoom(zoom)

            # 렌더 & 캡처
            vis.poll_events()
            vis.update_renderer()

            #os.makedirs(os.path.dirname(out_path), exist_ok=True)
            vis.capture_screen_image(out_path, do_render=True)
            vis.destroy_window()

        # 사용 예시
        scan_id = str(metainfo['scan_id'])  # 파일명에 쓰자
        #out_path = f"./outputs/occ_voxel_{scan_id}.png"
        out_path = f"occ_voxel_{scan_id}.png"
        save_voxelgrid_png(o3d_voxel_grid, out_path)
        print("saved ->", out_path)

        return

        shaped_voxels = voxels.unsqueeze(0).unsqueeze(0)

        #gt = data_sample.gt_occupancy.cpu().detach().unsqueeze(0)
        #gt = occ_multiscale_supervision(gt, 1,
        #                                shaped_voxels.shape,
        #                                None)
        #gt = gt.squeeze(0)
        #mask = (gt > 0) & (gt < 81)
        #assert voxels.shape == gt.shape

        grid_coords = get_grid_coords(
            [voxels.shape[0], voxels.shape[1], voxels.shape[2]], voxel_size
        )

        #grid_coords_gt = np.vstack([grid_coords.T, gt.reshape(-1)]).T
        #grid_coords_gt[grid_coords_gt[:, 3] == 17, 3] = 20

        grid_coords = np.vstack([grid_coords.T, voxels.reshape(-1)]).T
        grid_coords[grid_coords[:, 3] == 17, 3] = 0

        # Get the voxels inside FOV
        fov_grid_coords = grid_coords
        #fov_grid_coords_gt = grid_coords_gt

        # Remove empty and unknown voxels
        fov_voxels = fov_grid_coords[
            (fov_grid_coords[:, 3] > 0) & (fov_grid_coords[:, 3] < 12)
        ]
        voxel_size = sum(voxel_size) / 3
    
        COLORS = np.array([
            [ 22, 191, 206, 255], # 00 free
            [214,  38,  40, 255], # 01 ceiling
            [ 43, 160,  43, 255], # 02 floor
            [158, 216, 229, 255], # 03 wall
            [114, 158, 206, 255], # 04 window
            [204, 204,  91, 255], # 05 chair
            [255, 186, 119, 255], # 06 bed
            [147, 102, 188, 255], # 07 sofa
            [ 30, 119, 181, 255], # 08 table
            [188, 188,  33, 255], # 09 tvs
            [255, 127,  12, 255], # 10 furniture
            [196, 175, 214, 255], # 11 objects
            [153, 153, 153, 255], # 12 unknown
        ]).astype(np.uint8)
        
        occ_points = fov_voxels[:, :3]
        occ_labels = fov_voxels[:, 3]
        occ_colors = COLORS[occ_labels.astype(np.int32), :3].astype(np.float32) / 255.0
        o3d_pcd = o3d.geometry.PointCloud()
        o3d_pcd.points = o3d.utility.Vector3dVector(occ_points)
        o3d_pcd.colors = o3d.utility.Vector3dVector(occ_colors)
        o3d_voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(o3d_pcd, voxel_size=voxel_size)
        #o3d.visualization.draw_geometries([o3d_voxel_grid])

        def save_voxelgrid_png(o3d_voxel_grid,
                            out_path,
                            width=1600,
                            height=1200,
                            bg_color=(0, 0, 0),
                            front=(0.5, -0.5, -1.0),
                            lookat=None,
                            up=(0, -1, 0),
                            zoom=0.7):
            # 비가시 모드 창 생성
            vis = o3d.visualization.Visualizer()
            vis.create_window(visible=False, width=width, height=height)
            vis.add_geometry(o3d_voxel_grid)

            # 렌더 옵션
            opt = vis.get_render_option()
            opt.background_color = np.asarray(bg_color, dtype=np.float64)

            eye    = np.array([3.2, 3.2, 6.0], dtype=np.float64)
            lookat = np.array([3.2, 3.2, 0.0], dtype=np.float64)  # 정면(아래)으로 볼 지점
            front  = (lookat - eye); front = front / np.linalg.norm(front)  # (0,0,-1)로 됨
            up     = np.array([0.0, 1.0, 0.0], dtype=np.float64)  # 필요시 [0,-1,0]로 바꿔보면 느낌 달라짐

            # 카메라 뷰 세팅 (대충 중앙 바라보게)
            ctr = vis.get_view_control()
            if lookat is None:
                aabb = o3d_voxel_grid.get_axis_aligned_bounding_box()
                lookat = aabb.get_center()
            ctr.set_front(front)
            ctr.set_up(up)
            ctr.set_lookat(lookat)
            ctr.set_zoom(zoom)

            # 렌더 & 캡처
            vis.poll_events()
            vis.update_renderer()

            #os.makedirs(os.path.dirname(out_path), exist_ok=True)
            vis.capture_screen_image(out_path, do_render=True)
            vis.destroy_window()

        # 사용 예시
        scan_id = str(metainfo['scan_id'])  # 파일명에 쓰자
        #out_path = f"./outputs/occ_voxel_{scan_id}.png"
        out_path = f"occ_voxel_{scan_id}.png"
        save_voxelgrid_png(o3d_voxel_grid, out_path)
        print("saved ->", out_path)





         
