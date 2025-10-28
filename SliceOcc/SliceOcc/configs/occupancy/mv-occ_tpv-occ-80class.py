_base_ = ['../default_runtime.py']
n_points = 25600 #100000

# origin for multi-view scannet is set to 0.5
# -1.28~1.28 -> -0.78~1.78
point_cloud_range = [-3.2, -3.2, -0.78, 3.2, 3.2, 1.78]
#point_cloud_range = [-3.2, -3.2, -1.28, 3.2, 3.2, 1.28]

prior_generator = dict(type='AlignedAnchor3DRangeGenerator',
                       ranges=[[-3.2, -3.2, -1.28, 3.2, 3.2, 1.28]],
                       rotations=[.0])

_dim_ = 192
find_unused_parameters = False

track_running_stats = False
_num_cams_ = 10
tpv_h_ = 40
tpv_w_ = 40
tpv_z_ = 16
scale_h = 1
scale_w = 1
scale_z = 1
grid_size = [tpv_h_*scale_h, tpv_w_*scale_w, tpv_z_*scale_z]
num_points_in_pillar = [16, 64, 64]
num_points = [32, 128, 128]
nbr_class = 81

grid_size_occ = [40, 40, 16]
coarse_ratio = 2
sweeps_num = 10

model = dict(
    type='DenseTPVOccPredictor',
    use_valid_mask=False,
    use_xyz_feat=True,
    point_cloud_range=point_cloud_range,
    data_preprocessor=dict(type='Det3DDataPreprocessor',
                           mean=[123.675, 116.28, 103.53],
                           std=[58.395, 57.12, 57.375],
                           bgr_to_rgb=True,
                           pad_size_divisor=32),
    tpv_aggregator=dict(
        type='TPVAggregator_Occ',
        tpv_h=tpv_h_,
        tpv_w=tpv_w_,
        tpv_z=tpv_z_,
        grid_size_occ=grid_size_occ,
        coarse_ratio=coarse_ratio,
        loss_weight=[1,1,1,1],
        nbr_classes=nbr_class,
        in_dims=_dim_,
        hidden_dims=2*_dim_,
        out_dims=_dim_,
        scale_h=scale_h,
        scale_w=scale_w,
        scale_z=scale_z
    ),
    lidar_tokenizer=dict(
        type='CylinderEncoder_Occ',
        grid_size=grid_size,
        in_channels=3,
        out_channels=128,
        fea_compre=None,
        base_channels=128,
        split=[8,8,8],
        track_running_stats=track_running_stats,
    ),
    lidar_backbone=dict(
        type='Swin',
        embed_dims=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4,
        in_channels=128,
        patch_size=4,
        strides=[1,2,2,2],
        frozen_stages=-1,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.2,
        patch_norm=True,
        out_indices=[1,2,3],
        with_cp=False,
        convert_weights=True,
        init_cfg=dict(
          type='Pretrained',
          checkpoint='pretrain/swin_tiny_patch4_window7_224.pth'),
    ),
    lidar_neck=dict(
        type='GeneralizedLSSFPN',
        in_channels=[192, 384, 768],
        out_channels=_dim_,
        start_level=0,
        num_outs=3,
        norm_cfg=dict(
          type='BN2d',
          requires_grad=True,
          track_running_stats=track_running_stats),
        act_cfg=dict(
          type='ReLU',
          inplace=True),
        upsample_cfg=dict(
          mode='bilinear',
          align_corners=False),
    ),    
    neck_3d=dict(type='IndoorImVoxelNeck',
                 in_channels=81,
                 out_channels=128,
                 n_blocks=[1, 1, 1]),
    bbox_head=dict(
        type='ImVoxelOccHead',
        volume_h=[20, 10, 5],
        volume_w=[20, 10, 5],
        volume_z=[8, 4, 2],
        num_classes=81,  # TO Be changed
        in_channels=[128, 128, 128],
        use_semantic=True),
    prior_generator=prior_generator,
    n_voxels=[40, 40, 16],  # voxel_size=(.16, .16, .16)
    coord_type='DEPTH',
)

dataset_type = 'EmbodiedScanDataset'
data_root = 'data'
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
backend_args = None

train_pipeline = [
    dict(type='LoadAnnotations3D',
         with_occupancy=True,
         with_visible_occupancy_masks=True),
    dict(type='MultiViewPipeline',
         n_images=30,
         transforms=[
             dict(type='LoadImageFromFile', backend_args=backend_args),
             dict(type='LoadDepthFromFile', backend_args=backend_args),
             dict(type='ConvertRGBDToPoints', coord_type='CAMERA'),
             dict(type='PointSample', num_points=100000),
             dict(type='Resize', scale=(480, 480), keep_ratio=False)
         ]),
    dict(type='AggregateMultiViewPoints', coord_type='DEPTH'),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='PointSample', num_points=n_points),
    dict(type='ConstructMultiViewMasks'),
    dict(
        type='Pack3DDetInputs',
        keys=['img', 'points', 'gt_bboxes_3d', 'gt_labels_3d', 'gt_occupancy'])
]

test_pipeline = [
    dict(type='LoadAnnotations3D',
         with_occupancy=True,
         with_visible_occupancy_masks=True),
    dict(type='MultiViewPipeline',
         n_images=30,
         ordered=True,
         transforms=[
             dict(type='LoadImageFromFile', backend_args=backend_args),
             dict(type='LoadDepthFromFile', backend_args=backend_args),
             dict(type='ConvertRGBDToPoints', coord_type='CAMERA'),
             dict(type='PointSample', num_points=100000),
             dict(type='Resize', scale=(480, 480), keep_ratio=False)
         ]),
    dict(type='AggregateMultiViewPoints', coord_type='DEPTH'),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='PointSample', num_points=n_points),
    dict(type='ConstructMultiViewMasks'),
    dict(
        type='Pack3DDetInputs',
        keys=['img', 'points', 'gt_bboxes_3d', 'gt_labels_3d', 'gt_occupancy'])
]

train_dataloader = dict(batch_size=1,
                        num_workers=1,
                        persistent_workers=True,
                        sampler=dict(type='DefaultSampler', shuffle=True),
                        dataset=dict(type=dataset_type,
                                     data_root=data_root,
                                     ann_file='embodiedscan_infos_train.pkl',
                                     pipeline=train_pipeline,
                                     test_mode=False,
                                     filter_empty_gt=True,
                                     box_type_3d='Euler-Depth',
                                     metainfo=metainfo))

val_dataloader = dict(batch_size=1,
                      num_workers=1,
                      persistent_workers=True,
                      drop_last=False,
                      sampler=dict(type='DefaultSampler', shuffle=False),
                      dataset=dict(type=dataset_type,
                                   data_root=data_root,
                                   ann_file='embodiedscan_infos_val.pkl',
                                   pipeline=test_pipeline,
                                   test_mode=True,
                                   filter_empty_gt=True,
                                   box_type_3d='Euler-Depth',
                                   metainfo=metainfo))
test_dataloader = val_dataloader

val_evaluator = dict(type='OccupancyMetric')
test_evaluator = val_evaluator

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=24, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# optimizer
optim_wrapper = dict(type='OptimWrapper',
                     optimizer=dict(type='AdamW', lr=1e-4, weight_decay=0.01),
                     clip_grad=dict(max_norm=35., norm_type=2))
param_scheduler = dict(type='MultiStepLR',
                       begin=0,
                       end=24,
                       by_epoch=True,
                       milestones=[16, 22],
                       gamma=0.1)

custom_hooks = [dict(type='EmptyCacheHook', after_iter=True)]

# hooks
default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=4))

# runtime
find_unused_parameters = True  # only 1 of 4 FPN outputs is used
