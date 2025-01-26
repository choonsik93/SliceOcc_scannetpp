from embodiedscan.registry import MODELS
from embodiedscan.models.utils.grid_mask import GridMask
from typing import Dict, List, Optional, Union
import torch
from torch import Tensor
from embodiedscan.utils.typing_config import SampleList, ForwardResults

from mmengine.model import BaseModel

@MODELS.register_module()
class SurroundOcc(BaseModel):
    def __init__(self,
                use_grid_mask=False,
                pts_voxel_encoder=None,
                pts_middle_encoder=None,
                pts_fusion_layer=None,
                img_backbone=None,
                pts_backbone=None,
                img_neck=None,
                pts_neck=None,
                pts_bbox_head=None,
                img_roi_head=None,
                img_rpn_head=None,
                train_cfg=None,
                test_cfg=None,
                use_semantic=True,
                is_vis=False,
                data_preprocessor=None,
                **kwargs):

        super(SurroundOcc, self).__init__(data_preprocessor=data_preprocessor, **kwargs)
        if pts_voxel_encoder:
            self.pts_voxel_encoder = MODELS.build(pts_voxel_encoder)
        if pts_middle_encoder:
            self.pts_middle_encoder = MODELS.build(pts_middle_encoder)
        if pts_backbone:
            self.pts_backbone = MODELS.build(pts_backbone)
        if pts_fusion_layer:
            self.pts_fusion_layer = MODELS.build(pts_fusion_layer)
        if pts_neck is not None:
            self.pts_neck = MODELS.build(pts_neck)
        if pts_bbox_head:
            pts_train_cfg = train_cfg.pts if train_cfg else None
            pts_bbox_head.update(train_cfg=pts_train_cfg)
            pts_test_cfg = test_cfg.pts if test_cfg else None
            pts_bbox_head.update(test_cfg=pts_test_cfg)
            self.pts_bbox_head = MODELS.build(pts_bbox_head)

        if img_backbone:
            self.img_backbone = MODELS.build(img_backbone)
        if img_neck is not None:
            self.img_neck = MODELS.build(img_neck)
        if img_rpn_head is not None:
            self.img_rpn_head = MODELS.build(img_rpn_head)
        if img_roi_head is not None:
            self.img_roi_head = MODELS.build(img_roi_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.grid_mask = GridMask(
            True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7)
        # TODO: grid_mask
        self.use_grid_mask = use_grid_mask
        self.fp16_enabled = False
        self.use_semantic = use_semantic
        self.is_vis = is_vis
        
    def extract_img_feat(self, img, img_metas, len_queue=None):
        """Extract features of images."""
        B = img.size(0)
        if img is not None:
            # input_shape = img.shape[-2:]
            # # update real input shape of each single img
            # for img_meta in img_metas:
            #     img_meta.update(input_shape=input_shape)

            if img.dim() == 5 and img.size(0) == 1:
                img.squeeze_(0)
            elif img.dim() == 5 and img.size(0) > 1:
                B, N, C, H, W = img.size()
                img = img.reshape(B * N, C, H, W)
            if self.use_grid_mask:
                img = self.grid_mask(img)

            img_feats = self.img_backbone(img)
            if isinstance(img_feats, dict):
                img_feats = list(img_feats.values())
        else:
            return None
        if self.with_img_neck:
            img_feats = self.img_neck(img_feats)

        img_feats_reshaped = []
        for img_feat in img_feats:
            BN, C, H, W = img_feat.size()
            if len_queue is not None:
                img_feats_reshaped.append(img_feat.view(int(B/len_queue), len_queue, int(BN / B), C, H, W))
            else:
                img_feats_reshaped.append(img_feat.view(B, int(BN / B), C, H, W))
        return img_feats_reshaped

    def extract_feat(self, batch_inputs_dict: dict, batch_input_metas: List[dict], len_queue=None):
        img = batch_inputs_dict['imgs']
        batch_size = img.shape[0]
        
        if len(img.shape) == 5:
            B, N, C, H, W = img.shape
            img = img.reshape([-1] + list(img.shape)[2:])
            if self.use_grid_mask:
                img = self.grid_mask(img)
            img_feat = self.img_backbone(img)
            img_feat = self.img_neck(img_feat)
            img_feat_reshaped = [x.reshape([batch_size, -1] + list(x.shape)[1:]) for x in img_feat]
        else:
            if self.use_grid_mask:
                img = self.grid_mask(img)
            img_feat = self.img_backbone(img)
            img_feat = self.img_neck(img_feat)
            img_feat_reshaped = img_feat
        
        return img_feat_reshaped

    def forward_pts_train(self,
                          pts_feats,
                          gt_occ,
                          img_metas):

        outs = self.pts_bbox_head(
            pts_feats, img_metas)
        loss_inputs = [gt_occ, outs]
        losses = self.pts_bbox_head.loss(*loss_inputs, img_metas=img_metas)
        return losses

    def loss(self, batch_inputs_dict: Dict[List, torch.Tensor], batch_data_samples: SampleList, **kwargs) -> SampleList:
        batch_input_metas = [item.metainfo for item in batch_data_samples]
        img_feats = self.extract_feat(batch_inputs_dict, batch_input_metas)
        losses = self.pts_bbox_head.loss(img_feats, batch_data_samples, **kwargs)
        return losses
    
    def predict(self, batch_inputs_dict: Dict[str, Optional[Tensor]], batch_data_samples: SampleList, **kwargs) -> SampleList:
        batch_input_metas = [item.metainfo for item in batch_data_samples]
        img_feats = self.extract_feat(batch_inputs_dict, batch_input_metas)
        predict = self.pts_bbox_head.predict(img_feats, batch_data_samples)
        prediction = self.add_occupancy_to_data_sample(batch_data_samples, predict)
        # TODO: add add_pred_to_datasample
        return prediction
    
    def forward(self,
                inputs: Union[dict, List[dict]],
                data_samples: Optional[List] = None,
                mode: str = 'tensor',
                **kwargs) -> ForwardResults:
        """The unified entry for a forward process in both training and test.

        The method should accept three modes: "tensor", "predict" and "loss":

        - "tensor": Forward the whole network and return tensor or tuple of
        tensor without any post-processing, same as a common nn.Module.
        - "predict": Forward and return the predictions, which are fully
        processed to a list of :obj:`Det3DDataSample`.
        - "loss": Forward and return a dict of losses according to the given
        inputs and data samples.

        Note that this method doesn't handle neither back propagation nor
        optimizer updating, which are done in the :meth:`train_step`.

        Args:
            inputs  (dict | list[dict]): When it is a list[dict], the
                outer list indicate the test time augmentation. Each
                dict contains batch inputs
                which include 'points' and 'imgs' keys.

                - points (list[torch.Tensor]): Point cloud of each sample.
                - imgs (torch.Tensor): Image tensor has shape (B, C, H, W).
            data_samples (list[:obj:`Det3DDataSample`],
                list[list[:obj:`Det3DDataSample`]], optional): The
                annotation data of every samples. When it is a list[list], the
                outer list indicate the test time augmentation, and the
                inter list indicate the batch. Otherwise, the list simply
                indicate the batch. Defaults to None.
            mode (str): Return what kind of value. Defaults to 'tensor'.

        Returns:
            The return type depends on ``mode``.

            - If ``mode="tensor"``, return a tensor or a tuple of tensor.
            - If ``mode="predict"``, return a list of :obj:`Det3DDataSample`.
            - If ``mode="loss"``, return a dict of tensor.
        """
        if mode == 'loss':
            return self.loss(inputs, data_samples, **kwargs)
        elif mode == 'predict':
            return self.predict(inputs, data_samples, **kwargs)
        else:
            raise RuntimeError(f'Invalid mode "{mode}". '
                               'Only supports loss, predict and tensor mode')
    
    def add_occupancy_to_data_sample(self, data_samples: SampleList, pred):
        for i, data_sample in enumerate(data_samples):
            data_sample.pred_occupancy = pred[i]
        return data_samples

    def generate_output(self, pred_occ, img_metas):
        import open3d as o3d
        import os
        import numpy as np
        
        color_map = np.array(
                [
                    [0, 0, 0, 255],
                    [255, 120, 50, 255],  # barrier              orangey
                    [255, 192, 203, 255],  # bicycle              pink
                    [255, 255, 0, 255],  # bus                  yellow
                    [0, 150, 245, 255],  # car                  blue
                    [0, 255, 255, 255],  # construction_vehicle cyan
                    [200, 180, 0, 255],  # motorcycle           dark orange
                    [255, 0, 0, 255],  # pedestrian           red
                    [255, 240, 150, 255],  # traffic_cone         light yellow
                    [135, 60, 0, 255],  # trailer              brown
                    [160, 32, 240, 255],  # truck                purple
                    [255, 0, 255, 255],  # driveable_surface    dark pink
                    # [175,   0,  75, 255],       # other_flat           dark red
                    [139, 137, 137, 255],
                    [75, 0, 75, 255],  # sidewalk             dard purple
                    [150, 240, 80, 255],  # terrain              light green
                    [230, 230, 250, 255],  # manmade              white
                    [0, 175, 0, 255],  # vegetation           green
                ]
            )
        
        
        if self.use_semantic:
            _, voxel = torch.max(torch.softmax(pred_occ, dim=1), dim=1)
        else:
            voxel = torch.sigmoid(pred_occ[:, 0])
        
        for i in range(voxel.shape[0]):
            x = torch.linspace(0, voxel[i].shape[0] - 1, voxel[i].shape[0])
            y = torch.linspace(0, voxel[i].shape[1] - 1, voxel[i].shape[1])
            z = torch.linspace(0, voxel[i].shape[2] - 1, voxel[i].shape[2])
            X, Y, Z = torch.meshgrid(x, y, z)
            vv = torch.stack([X, Y, Z], dim=-1).to(voxel.device)
        
            vertices = vv[voxel[i] > 0.5]
            vertices[:, 0] = (vertices[:, 0] + 0.5) * (img_metas[i]['pc_range'][3] - img_metas[i]['pc_range'][0]) /  img_metas[i]['occ_size'][0]  + img_metas[i]['pc_range'][0]
            vertices[:, 1] = (vertices[:, 1] + 0.5) * (img_metas[i]['pc_range'][4] - img_metas[i]['pc_range'][1]) /  img_metas[i]['occ_size'][1]  + img_metas[i]['pc_range'][1]
            vertices[:, 2] = (vertices[:, 2] + 0.5) * (img_metas[i]['pc_range'][5] - img_metas[i]['pc_range'][2]) /  img_metas[i]['occ_size'][2]  + img_metas[i]['pc_range'][2]
            
            vertices = vertices.cpu().numpy()
    
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(vertices)
            if self.use_semantic:
                semantics = voxel[i][voxel[i] > 0].cpu().numpy()
                color = color_map[semantics] / 255.0
                pcd.colors = o3d.utility.Vector3dVector(color[..., :3])
                vertices = np.concatenate([vertices, semantics[:, None]], axis=-1)
    
            save_dir = os.path.join('visual_dir', img_metas[i]['occ_path'].replace('.npy', '').split('/')[-1])
            os.makedirs(save_dir, exist_ok=True)


            o3d.io.write_point_cloud(os.path.join(save_dir, 'pred.ply'), pcd)
            np.save(os.path.join(save_dir, 'pred.npy'), vertices)
            for cam_id, cam_path in enumerate(img_metas[i]['filename']):
                os.system('cp {} {}/{}.jpg'.format(cam_path, save_dir, cam_id))