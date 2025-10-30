# Copyright (c) OpenRobotLab. All rights reserved.
from typing import List, Optional, Tuple, Union

import torch
from mmengine.structures import InstanceData

try:
    import MinkowskiEngine as ME
except ImportError:
    # Please follow getting_started.md to install MinkowskiEngine.
    ME = None
    pass

from mmengine.model import BaseModel

from embodiedscan.registry import MODELS, TASK_UTILS
from embodiedscan.structures.bbox_3d import get_proj_mat_by_coord_type
from embodiedscan.utils import ConfigType, OptConfigType
from embodiedscan.utils.typing_config import (ForwardResults, InstanceList,
                                              SampleList)

from ..layers.fusion_layers.point_fusion import (batch_point_sample, Dense3DKeyPointsGenerator,
                                                 point_sample)


@MODELS.register_module()
class DenseSliceOccPredictor(BaseModel):
    """Dense Fusion framework for occupancy prediction.

    Args:
        backbone (:obj:`ConfigDict` or dict): The image backbone config.
        backbone_3d (:obj:`ConfigDict` or dict): The 3D backbone config.
        neck (:obj:`ConfigDict` or dict): The image neck config.
        neck_3d (:obj:`ConfigDict` or dict): The 3D neck config.
        bbox_head (:obj:`ConfigDict` or dict): The bbox head config.
        prior_generator (:obj:`ConfigDict` or dict): The prior grid generator
            config.
        n_voxels (list): Number of voxels along x, y, z axis.
        coord_type (str): The type of coordinates of points cloud:
            'DEPTH', 'LIDAR', or 'CAMERA'.
        use_valid_mask (bool): Whether to use valid masks to handle
            visible voxels. Defaults to False.
        use_xyz_feat (bool): Whether to use xyz features.
            Defaults to False.
        point_cloud_range (list]): Point cloud range, [x_min, y_min, z_min,
            x_max, y_max, z_max], e.g., [-3.2, -3.2, -0.78, 3.2, 3.2, 1.78].
        train_cfg (:obj:`ConfigDict` or dict, optional): Config dict of
            training hyper-parameters. Defaults to None.
        test_cfg (:obj:`ConfigDict` or dict, optional): Config dict of test
            hyper-parameters. Defaults to None.
        data_preprocessor (dict or ConfigDict, optional): The pre-process
            config of :class:`BaseDataPreprocessor`.  it usually includes,
                ``pad_size_divisor``, ``pad_value``, ``mean`` and ``std``.
        init_cfg (:obj:`ConfigDict` or dict, optional): The initialization
            config. Defaults to None.
    """

    def __init__(self,
                 backbone: ConfigType,
                 #backbone_3d: ConfigType,
                 neck: ConfigType,
                 neck_3d: ConfigType,
                 slicenet: ConfigType,
                 bbox_head: ConfigType,
                 prior_generator: ConfigType,                                                                 
                 n_voxels: List,
                 n_anchors: List,
                 coord_type: str,
                 use_valid_mask=True,
                 use_xyz_feat: bool = False,
                 point_cloud_range=None,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptConfigType = None):
        super().__init__(data_preprocessor=data_preprocessor,
                         init_cfg=init_cfg)
        self.backbone = MODELS.build(backbone)
        #backbone_checkpoint_path = '/mnt/data/ljn/code/EmbodiedScan/work_dirs/mv-occ_8xb1_sliceformer-occ-80class/res50.pth'
        #backbone_checkpoint = torch.load(backbone_checkpoint_path)        
        #self.backbone.load_state_dict(backbone_checkpoint['state_dict'], strict=False)
        if neck is not None:
            self.neck = MODELS.build(neck)
        if neck_3d is not None:
            self.neck_3d = MODELS.build(neck_3d)
        if slicenet is not None:
            self.slicenet = MODELS.build(slicenet)
        bbox_head.update(train_cfg=train_cfg)
        bbox_head.update(test_cfg=test_cfg)
        self.bbox_head = MODELS.build(bbox_head)
        self.n_voxels = n_voxels
        self.n_anchors = n_anchors
        self.point_cloud_range = point_cloud_range
        prior_range = prior_generator['ranges'][0]
        self.voxel_stride = 1
        self.voxel_size = [(prior_range[3] - prior_range[0]) /
                           self.n_voxels[0] / self.voxel_stride,
                           (prior_range[4] - prior_range[1]) /
                           self.n_voxels[1] / self.voxel_stride,
                           (prior_range[5] - prior_range[2]) /
                           self.n_voxels[2] / self.voxel_stride]

        self.prior_generator = TASK_UTILS.build(prior_generator)

        self.coord_type = coord_type
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.use_valid_mask = use_valid_mask
        self.use_xyz_feat = use_xyz_feat

        if ME is None:
            raise ImportError(
                'Please follow `getting_started.md` to install MinkowskiEngine.`'  # noqa: E501
            )

    @property
    def with_neck(self):
        """Whether the detector has a 2D backbone."""
        return hasattr(self, 'neck') and self.neck is not None

    @property
    def with_neck_3d(self):
        """Whether the detector has a 3D neck."""
        return hasattr(self, 'neck_3d') and self.neck_3d is not None

    def extract_feat(self, batch_inputs_dict: dict,
                     batch_data_samples: SampleList):
        """Extract 3d features from the backbone -> fpn -> 3d projection.

        -> 3d neck -> bbox_head.

        Args:
            batch_inputs_dict (dict): The model input dict which include
                the 'imgs' key.

                    - imgs (torch.Tensor, optional): Image of each sample.
            batch_data_samples (list[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.

        Returns:
            Tuple:
             - torch.Tensor: Features of shape (N, C_out, N_x, N_y, N_z).
             - torch.Tensor: Valid mask of shape (N, 1, N_x, N_y, N_z).
        """
        # 1. Extract the feature volume from images
        if 'img' in batch_inputs_dict:
            img = batch_inputs_dict['img'].unsqueeze(0)
            batch_img_metas = [
            data_samples for data_samples in batch_data_samples
        ]
        else:
            img = batch_inputs_dict['imgs']
            batch_img_metas = [
                data_samples.metainfo for data_samples in batch_data_samples
            ]


        batch_size = img.shape[0]
        B, N, C, H, W = img.size()
        if len(img.shape) > 4:  # (B, n_views, C, H, W)
            img = img.reshape([-1] + list(img.shape)[2:])
            x = self.backbone(img)
            x = self.neck(x)[0]
            x = x.reshape([batch_size, -1] + list(x.shape)[1:])
        else:
            x = self.backbone(img)
            x = self.neck(x)[0]

        img_feats_reshaped = []
        for img_feat in x:
            BN, C, H, W = img_feat.size()
            img_feats_reshaped.append(img_feat.view(B, int(BN / B), C, H, W))
            
        prior_points = self.prior_generator.grid_anchors(
            [self.n_anchors[::-1]], device=img.device)[0][:, :3]


        if 'origin' in batch_img_metas[0]['depth2img'].keys():
            assert len(batch_img_metas) == 1, 'only support batch_size=1 here'
            prior_points += prior_points.new_tensor(
                batch_img_metas[0]['depth2img']['origin']).to(prior_points.device)
            # For calibration with original ImVoxelNet implementation
            # prior_points += prior_points.new_tensor([-0.08, -0.08, -0.08])

        volumes, valid_preds = [], []
        for feature, img_meta in zip(x, batch_img_metas):
            img_scale_factor = (prior_points.new_tensor(
                img_meta['scale_factor'][:2])
                                if 'scale_factor' in img_meta.keys() else 1)
            img_flip = img_meta['flip'] if 'flip' in img_meta.keys() else False
            img_crop_offset = (prior_points.new_tensor(
                img_meta['img_crop_offset'])
                               if 'img_crop_offset' in img_meta.keys() else 0)
            proj_mat = get_proj_mat_by_coord_type(img_meta, self.coord_type)
            # Multi-View ImVoxelNet
            if isinstance(proj_mat, dict):
                assert 'extrinsic' in proj_mat.keys()
                assert 'intrinsic' in proj_mat.keys()
                projection = []
                # Support different intrinsic matrices for different images
                # if the original intrinsic is only a matrix
                # we will simply copy it to construct the intrinsic matrix list
                # in MultiViewPipeline
                assert isinstance(proj_mat['intrinsic'], list)
                for proj_idx in range(len(proj_mat['extrinsic'])):
                    intrinsic = img.new_tensor(proj_mat['intrinsic'][proj_idx])
                    extrinsic = img.new_tensor(proj_mat['extrinsic'][proj_idx])
                    projection.append(intrinsic @ extrinsic)
                proj_mat = torch.stack(projection)
                volume = batch_point_sample(
                    img_meta,
                    img_features=feature,
                    points=prior_points,
                    proj_mat=proj_mat,
                    coord_type=self.coord_type,
                    img_scale_factor=img_scale_factor,
                    img_crop_offset=img_crop_offset,
                    img_flip=img_flip,
                    img_pad_shape=img.shape[-2:],
                    img_shape=img_meta['img_shape'][:2],
                    aligned=False)
            else:
                proj_mat = prior_points.new_tensor(proj_mat)
                volume = point_sample(img_meta,
                                      img_features=feature[None, ...],
                                      points=prior_points,
                                      proj_mat=proj_mat,
                                      coord_type=self.coord_type,
                                      img_scale_factor=img_scale_factor,
                                      img_crop_offset=img_crop_offset,
                                      img_flip=img_flip,
                                      img_pad_shape=img.shape[-2:],
                                      img_shape=img_meta['img_shape'][:2],
                                      aligned=False)
            volumes.append(
                volume.reshape(self.n_anchors[::-1] + [-1]).permute(3, 2, 1, 0))
            valid_preds.append(
                ~torch.all(volumes[-1] == 0, dim=0, keepdim=True))
        img_volume = torch.stack(volumes).contiguous()
        del volumes
        '''
        volumes, valid_preds = [], []
        
        key_generator = Dense3DKeyPointsGenerator(pc_range=[-3.2, -3.2, -1.28, 3.2, 3.2, 1.28]).to(prior_points.device)
        key_points = key_generator(prior_points, img_volume)
        assert key_points.shape == prior_points.shape

        volume = batch_point_sample(
                    img_meta,
                    img_features=feature,
                    points=key_points ,
                    proj_mat=proj_mat,
                    coord_type=self.coord_type,
                    img_scale_factor=img_scale_factor,
                    img_crop_offset=img_crop_offset,
                    img_flip=img_flip,
                    img_pad_shape=img.shape[-2:],
                    img_shape=img_meta['img_shape'][:2],
                    aligned=False)
        volumes.append(
                volume.reshape(self.n_anchors[::-1] + [-1]).permute(3, 2, 1, 0))
        valid_preds.append(
                ~torch.all(volumes[-1] == 0, dim=0, keepdim=True))
        img_volume = torch.stack(volumes).contiguous()
        '''
        '''
        points = batch_inputs_dict['points']
        points_batch = []
        grid_ind_batch = []
        for p in points:
            if p.size(0) == 25600:
                points_batch.append(p)
                grid_ind = (p[:, :3] - torch.FloatTensor(self.point_cloud_range[:3]).to(p.device) - torch.FloatTensor([0.002, 0.002, 0.002]).to(p.device)) / torch.FloatTensor(self.voxel_size).to(p.device)
                grid_ind = torch.floor(grid_ind).to(torch.int32)
                grid_ind_batch.append(grid_ind)
            else:
                points_batch.append(points[0])
                grid_ind = (points[0][:, :3] - torch.FloatTensor(self.point_cloud_range[:3]).to(p.device) - torch.FloatTensor([0.001, 0.001, 0.001]).to(p.device)) / torch.FloatTensor(self.voxel_size).to(p.device)
                grid_ind = torch.floor(grid_ind).to(torch.int32)
                grid_ind_batch.append(grid_ind)

        '''
        #print(img_volume.shape)
        p2i_fusion_feature = self.slicenet(img_feats_reshaped, batch_img_metas, img_volume)
        x = p2i_fusion_feature
        x = self.neck_3d(x)

        return x, torch.stack(valid_preds).float()

    def loss(self, batch_inputs_dict: dict, batch_data_samples: SampleList,
             **kwargs) -> Union[dict, list]:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            batch_inputs_dict (dict): The model input dict which include
                the 'imgs' key.

                    - imgs (torch.Tensor, optional): Image of each sample.
            batch_data_samples (list[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.

        Returns:
            dict: A dictionary of loss components.
        """
        x, valid_preds = self.extract_feat(batch_inputs_dict,
                                           batch_data_samples)
        #
        # For indoor datasets ImVoxelNet uses ImVoxelHead that handles
        # mask of visible voxels.
        
        if self.coord_type in ('DEPTH', 'CAMERA') and self.use_valid_mask:
            x += (valid_preds, )

        losses = self.bbox_head.loss(x, batch_data_samples, **kwargs)

        return losses

    def predict(self, batch_inputs_dict: dict, batch_data_samples: SampleList,
                **kwargs) -> SampleList:
        """Predict results from a batch of inputs and data samples with post-
        processing.

        Args:
            batch_inputs_dict (dict): The model input dict which include
                the 'imgs' key.

                    - imgs (torch.Tensor, optional): Image of each sample.

            batch_data_samples (List[:obj:`Det3DDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance_3d`, `gt_panoptic_seg_3d` and `gt_sem_seg_3d`.

        Returns:
            list[:obj:`Det3DDataSample`]: Detection results of the
            input images. Each Det3DDataSample usually contain
            'pred_instances_3d'. And the ``pred_instances_3d`` usually
            contains following keys.

                - scores_3d (Tensor): Classification scores, has a shape
                    (num_instance, )
                - labels_3d (Tensor): Labels of bboxes, has a shape
                    (num_instances, ).
                - bboxes_3d (Tensor): Contains a tensor with shape
                    (num_instances, C) where C >=7.
        """
        #print(batch_inputs_dict.keys())
        x, valid_preds = self.extract_feat(batch_inputs_dict,
                                           batch_data_samples)
        #  
        # For indoor datasets ImVoxelNet uses ImVoxelHead that handles
        # mask of visible voxels.
        if self.coord_type in ('DEPTH', 'CAMERA') and self.use_valid_mask:
            x += (valid_preds, )

        results_list = self.bbox_head.predict(x, batch_data_samples, **kwargs)
        #print(type(batch_data_samples))
        predictions = self.add_occupancy_to_data_sample(
            batch_data_samples, results_list)

        from embodiedscan.visualizer import EmbodiedScanBaseVisualizer
        visualizer = EmbodiedScanBaseVisualizer.get_current_instance()
        visualizer.visualize_occupancy(predictions)
        
        return predictions

    def add_occupancy_to_data_sample(self, data_samples: SampleList, pred):
        for i, data_sample in enumerate(data_samples):
            data_sample.pred_occupancy = pred[i]
            #data_sample["pred_occupancy"] = pred[i]
        return data_samples

    def _forward(self, batch_inputs_dict: dict, batch_data_samples: SampleList,
                 *args, **kwargs) -> Tuple[List[torch.Tensor]]:
        """Network forward process. Usually includes backbone, neck and head
        forward without any post-processing.

        Args:
            batch_inputs_dict (dict): The model input dict which include
                the 'imgs' key.

                    - imgs (torch.Tensor, optional): Image of each sample.
            batch_data_samples (List[:obj:`Det3DDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance_3d`, `gt_panoptic_seg_3d` and `gt_sem_seg_3d`.

        Returns:
            tuple[list]: A tuple of features from ``bbox_head`` forward.
        """
        x, valid_preds = self.extract_feat(batch_inputs_dict,
                                           batch_data_samples)
        # For indoor datasets ImVoxelNet uses ImVoxelHead that handles
        # mask of visible voxels.
        if self.coord_type in ('DEPTH', 'CAMERA') and self.use_valid_mask:
            x += (valid_preds, )
        results = self.bbox_head.forward(x)
        return results

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
            return self.predict(inputs, data_samples, **kwargs)
            #raise RuntimeError(f'Invalid mode "{mode}". '
            #                   'Only supports loss, predict and tensor mode')

    def add_pred_to_datasample(
        self,
        data_samples: SampleList,
        data_instances_3d: Optional[InstanceList] = None,
        data_instances_2d: Optional[InstanceList] = None,
    ) -> SampleList:
        """Convert results list to `Det3DDataSample`.

        Subclasses could override it to be compatible for some multi-modality
        3D detectors.

        Args:
            data_samples (list[:obj:`Det3DDataSample`]): The input data.
            data_instances_3d (list[:obj:`InstanceData`], optional): 3D
                Detection results of each sample.
            data_instances_2d (list[:obj:`InstanceData`], optional): 2D
                Detection results of each sample.

        Returns:
            list[:obj:`Det3DDataSample`]: Detection results of the
            input. Each Det3DDataSample usually contains
            'pred_instances_3d'. And the ``pred_instances_3d`` normally
            contains following keys.

            - scores_3d (Tensor): Classification scores, has a shape
              (num_instance, )
            - labels_3d (Tensor): Labels of 3D bboxes, has a shape
              (num_instances, ).
            - bboxes_3d (Tensor): Contains a tensor with shape
              (num_instances, C) where C >=7.

            When there are image prediction in some models, it should
            contains  `pred_instances`, And the ``pred_instances`` normally
            contains following keys.

            - scores (Tensor): Classification scores of image, has a shape
              (num_instance, )
            - labels (Tensor): Predict Labels of 2D bboxes, has a shape
              (num_instances, ).
            - bboxes (Tensor): Contains a tensor with shape
              (num_instances, 4).
        """

        assert (data_instances_2d is not None) or \
               (data_instances_3d is not None),\
               'please pass at least one type of data_samples'

        if data_instances_2d is None:
            data_instances_2d = [
                InstanceData() for _ in range(len(data_instances_3d))
            ]
        if data_instances_3d is None:
            data_instances_3d = [
                InstanceData() for _ in range(len(data_instances_2d))
            ]

        for i, data_sample in enumerate(data_samples):
            data_sample.pred_instances_3d = data_instances_3d[i]
            data_sample.pred_instances = data_instances_2d[i]
        return data_samples
