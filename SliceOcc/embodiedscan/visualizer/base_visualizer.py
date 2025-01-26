import os

from mmengine.dist import master_only
from mmengine.visualization import Visualizer

from embodiedscan.registry import VISUALIZERS

try:
    import open3d as o3d

    from embodiedscan.visualization.utils import _9dof_to_box, nms_filter
except ImportError:
    o3d = None

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
    
    

    '''
    @master_only
    def visualize_occupancy(self,
                        data_samples,
                        voxel_size=[0.2,0.2,0.2],  # voxel size in the real world
                        ):

        assert len(data_samples) == 1
        data_sample = data_samples[0]

        metainfo = data_sample.metainfo
        
        root_dir = self.get_root_dir(metainfo['img_path'][0])
        print(root_dir, metainfo['scan_id'])
        ply_file = self.get_ply(root_dir, metainfo['scan_id'])
        axis_align_matrix = metainfo['axis_align_matrix']

        # Compute the voxels coordinates
        voxels = data_sample.pred_occupancy.cpu().detach()


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


        figure = mlab.figure(size=(2560, 1440), bgcolor=(1, 1, 1))
        # Draw occupied inside FOV voxels
        voxel_size = sum(voxel_size) / 3
 
        
        plt_plot_fov = mlab.points3d(
            fov_voxels[:, 1],
            fov_voxels[:, 0],
            fov_voxels[:, 2],
            fov_voxels[:, 3],
            colormap="viridis",
            scale_factor=0.95 * voxel_size,
            mode="cube",
            opacity=1.0,
            vmin=1,
            vmax=19, # 16
        )
    
        

        colors = np.array(
            [
                [255, 120,  50, 255],       # floor                orange
                [255, 192, 203, 255],       # wall                 pink
                [255, 255,   0, 255],       # chair                yellow
                [  0, 150, 245, 255],       # cabinet              blue
                [  0, 255, 255, 255],       # door                 cyan
                [0, 175,   0, 255],         # table                green
                [255,   0,   0, 255],       # couch                red
                [127, 127, 127, 255],       # shelf                 gray
                [135,  60,   0, 255],       # window              brown
                [160,  32, 240, 255],       # bed                purple                
                [255,   0, 255, 255],       # driveable_surface    dark pink
                [175,   0,  75, 255],       # other_flat           dark red
                [139, 137, 137, 255],
                [ 75,   0,  75, 255],       # sidewalk             dard purple
                [150, 240,  80, 255],       # terrain              light green          
                [230, 230, 250, 255],       # manmade              white
                [  0, 175,   0, 255],       # vegetation           green
                [  0, 255, 127, 255],       # ego car              dark cyan
                [255,  99,  71, 255],       # ego car
                [  0, 191, 255, 255],        # ego car              
            ]
        ).astype(np.uint8)
        
        plt_plot_fov.glyph.scale_mode = "scale_by_vector"
        plt_plot_fov.module_manager.scalar_lut_manager.lut.table = colors


        scene = figure.scene
        
        scene.camera.position = [  0.75131739, -35.08337438,  16.71378558]
        scene.camera.focal_point = [  0.75131739, -34.21734897,  16.21378558]      
        scene.camera.view_angle = 40.0
        scene.camera.view_up = [0., 1., 0.]
        scene.camera.clipping_range = [0.01, 200.]
        

        #scene.camera.position = [0, 0, 50]
        #scene.camera.focal_point = [0, 0, 0]
        #scene.camera.view_angle = 40.0
        #scene.camera.view_up = [0, 1, 0]
        #scene.camera.clipping_range = [0.01, 200]

        scene.camera.compute_view_plane_normal()
        scene.render()
        save_dir = "/mnt/data/ljn/code/EmbodiedScan/vis/slice8/"
        scene_id = metainfo['scan_id']    
        scene_id = scene_id.replace("/", "_")  
        mlab.savefig(os.path.join(save_dir, f'vis_{scene_id}.png'))   

        mlab.close()
    
    '''



         
