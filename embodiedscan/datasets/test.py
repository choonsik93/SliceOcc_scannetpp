import os
import warnings
from typing import Callable, List, Optional, Union

import mmengine
import numpy as np
from mmengine.dataset import BaseDataset
from mmengine.fileio import load

from embodiedscan.registry import DATASETS, TRANSFORMS
from embodiedscan.structures import get_box_type
from mmengine.config import Config
from copy import deepcopy
from mmengine.dataset import Compose



if __name__ == "__main__":

    _base_ = ['../default_runtime.py']
    n_points = 100000

    # origin for multi-view scannet is set to 0.5
    # -1.28~1.28 -> -0.78~1.78
    point_cloud_range = [-3.2, -3.2, -0.78, 3.2, 3.2, 1.78]

    prior_generator = dict(type='AlignedAnchor3DRangeGenerator',
                        ranges=[[-3.2, -3.2, -1.28, 3.2, 3.2, 1.28]],
                        rotations=[.0])

    backend_args = None
    train_pipeline = [
        dict(type='LoadAnnotations3D',
            with_occupancy=True,
            with_visible_occupancy_masks=True,
            with_visible_instance_masks=True),
        dict(type='MultiViewPipeline',
            n_images=10,
            transforms=[
                dict(type='LoadImageFromFile', backend_args=backend_args),
                dict(type='LoadDepthFromFile', backend_args=backend_args),
                dict(type='ConvertRGBDToPoints', coord_type='CAMERA'),
                dict(type='PointSample', num_points=n_points // 10),
                dict(type='Resize', scale=(480, 480), keep_ratio=False)
            ]),
        dict(type='AggregateMultiViewPoints', coord_type='DEPTH',
            save_slices=True),
        dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
        # dict(type='PointSample', num_points=n_points),
        dict(type='ConstructMultiSweeps'),
        dict(
            type='Pack3DDetInputs',
            keys=['img', 'points', 'gt_bboxes_3d', 'gt_labels_3d', 'gt_occupancy'])
    ]

    class_names = ('floor', 'wall', 'chair', 'cabinet', 'door', 'table', 'couch',
                'shelf', 'window', 'bed', 'curtain', 'desk', 'doorframe',
                'plant', 'stairs', 'pillow', 'wardrobe', 'picture', 'bathtub',
                'box', 'counter', 'bench', 'stand', 'rail', 'sink', 'clothes',
                'mirror', 'toilet', 'refrigerator', 'lamp', 'book', 'dresser',
                'stool', 'fireplace', 'tv', 'blanket', 'commode',
                'washing machine', 'monitor', 'window frame', 'radiator', 'mat',
                'shower', 'rack', 'towel', 'ottoman', 'column', 'blinds',
                'stove', 'bar', 'pillar', 'bin', 'heater', 'clothes dryer',
                'backpack', 'blackboard', 'decoration', 'roof', 'bag', 'steps',
                'windowsill', 'cushion', 'carpet', 'copier', 'board',
                'countertop', 'basket', 'mailbox', 'kitchen island',
                'washbasin', 'bicycle', 'drawer', 'oven', 'piano',
                'excercise equipment', 'beam', 'partition', 'printer',
                'microwave', 'frame')

    metainfo = dict(classes=class_names,
                    occ_classes=class_names,
                    box_type_3d='euler-depth')

    config = Config.fromfile("/mnt/data/ljn/code/EmbodiedScan/configs/occupancy/mv-occ_tpvformer-occ-80class.py")
    
    train_pipeline = Compose(config.train_pipeline)
    pipelines = []
    for t in config.train_pipeline:
        train_pipeline = TRANSFORMS.build(config.train_pipeline) #deepcopy(config.train_pipeline)
        pipelines.append(train_pipeline)
    
    #dataset = DATASETS.build(config.train_dataloader.dataset)