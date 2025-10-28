_base_ = ['../default_runtime.py']

point_cloud_range = [-3.2, -3.2, -1.28, 3.2, 3.2, 1.28]
occ_size = [40, 40, 16]
use_semantic = True

_dim_ = [128, 256, 512]
_ffn_dim_ = [256, 512, 1024]
volume_h_ = [20, 10, 5]
volume_w_ = [20, 10, 5]
volume_z_ = [8, 4, 2]
_num_points_ = [2, 4, 8]
_num_layers_ = [1, 3, 6]

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

model = dict(
    type='SurroundOcc',
    use_grid_mask=True,
    use_semantic=use_semantic,
    is_vis=True,
    data_preprocessor=dict(
        type='Det3DDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=32),
    img_backbone=dict(
        type='mmdet.ResNet',
        depth=50,
        num_stages=4,
        out_indices=(1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50'),
        style='pytorch'),
    img_neck=dict(
        type='mmdet.FPN',
        in_channels=[512, 1024, 2048],
        out_channels=512,
        start_level=0,
        add_extra_convs='on_output',
        num_outs=3,
        relu_before_extra_convs=True),
    pts_bbox_head=dict(
        type='SurroundOccHead',
        volume_h=volume_h_,
        volume_w=volume_w_,
        volume_z=volume_z_,
        num_query=900,
        num_classes=len(class_names) + 1,     # TO Be changed
        conv_input=[_dim_[2], 256, _dim_[1], 128, _dim_[0], 64, 64],
        conv_output=[256, _dim_[1], 128, _dim_[0], 64, 64, 32],
        out_indices=[0, 2, 4, 6],
        upsample_strides=[1,2,1,2,1,2,1],
        embed_dims=_dim_,
        img_channels=[512, 512, 512],
        use_semantic=use_semantic,
        transformer_template=dict(
            type='Detr3DTransformer',
            num_cams=20,    # TO Be changed
            embed_dims=_dim_,
            encoder=dict(
                type='OccEncoder',
                num_layers=_num_layers_,
                pc_range=point_cloud_range,
                return_intermediate=False,
                transformerlayers=dict(
                    type='OccLayer',
                    attn_cfgs=[
                        dict(
                            type='SpatialCrossAttention',
                            pc_range=point_cloud_range,
                            num_cams=20, # TO Be Changed
                            deformable_attention=dict(
                                type='MSDeformableAttention3D',
                                embed_dims=_dim_,
                                num_points=_num_points_,
                                num_levels=1),
                            embed_dims=_dim_,
                        )
                    ],
                    feedforward_channels=_ffn_dim_,
                    ffn_dropout=0.1,
                    embed_dims=_dim_,
                    conv_num=2,
                    operation_order=('cross_attn', 'norm', 'ffn', 'norm', 'conv') )
                )
            ),
    )
)

train_pipeline = [
    dict(type='LoadAnnotations3D',
         with_occupancy=True,
         with_visible_occupancy_masks=True),
    dict(
        type='MultiViewPipeline',
        n_images=20,
        transforms=[
            dict(type='LoadImageFromFile', backend_args=backend_args),
            dict(type='Resize', scale=(640, 480), keep_ratio=True)
        ]),
    # dict(type='RandomShiftOrigin', std=(.7, .7, .0)),
    dict(type='ConstructMultiViewMasks'),
    dict(type='Pack3DDetInputs', keys=['img', 'gt_bboxes_3d', 'gt_labels_3d', 'gt_occupancy'])
]

test_pipeline = [
    dict(type='LoadAnnotations3D',
         with_occupancy=True,
         with_visible_occupancy_masks=True),
    dict(
        type='MultiViewPipeline',
        n_images=20,
        transforms=[
            dict(type='LoadImageFromFile', backend_args=backend_args),
            dict(type='Resize', scale=(640, 480), keep_ratio=True)
        ]),
    dict(type='ConstructMultiViewMasks'),
    dict(type='Pack3DDetInputs', keys=['img', 'gt_bboxes_3d', 'gt_labels_3d', 'gt_occupancy'])
]

train_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='embodiedscan_infos_train.pkl', #'scannet_infos_train_newann.pkl',       # 'scannet_infos_train_debug.pkl',
        pipeline=train_pipeline,
        test_mode=False,
        filter_empty_gt=True,
        box_type_3d='Euler-Depth',
        metainfo=metainfo))

val_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='embodiedscan_infos_val.pkl',#'scannet_infos_val_newann.pkl',               #'debug_test.pkl',
        pipeline=test_pipeline,
        test_mode=True,
        filter_empty_gt=True,
        box_type_3d='Euler-Depth',
        metainfo=metainfo))
test_dataloader = val_dataloader

val_evaluator = dict(type='OccupancyMetric')
test_evaluator = val_evaluator

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=48, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=2e-4, weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys={'img_backbone': dict(lr_mult=0.1, decay_mult=1.0)}),
    clip_grad=dict(max_norm=35., norm_type=2))
param_scheduler = [
    dict(
        type='MultiStepLR',
        begin=0,
        end=48,
        by_epoch=True,
        milestones=[32, 44],
        gamma=0.1)
]

# hooks
default_hooks = dict(checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=1))

# runtime
find_unused_parameters = True  # only 1 of 4 FPN outputs is used