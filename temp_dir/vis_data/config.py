_dim_ = 256
_ffn_dim_ = 512
_num_cams_ = 20
_num_levels_ = 4
_pos_dim_ = 128
anchor_z_ = 32
backend_args = None
cam_point_range = [
    -3.2,
    -3.2,
    -1.28,
    3.2,
    3.2,
    1.28,
]
class_names = (
    'ceiling',
    'floor',
    'wall',
    'window',
    'chair',
    'bed',
    'sofa',
    'table',
    'tv',
    'furniture',
    'objects',
)
custom_hooks = [
    dict(after_iter=True, type='EmptyCacheHook'),
]
data_root = '/data'
dataset_type = 'ScannetppDataset'
default_hooks = dict(
    checkpoint=dict(interval=32, max_keep_ckpts=32, type='CheckpointHook'),
    logger=dict(interval=50, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'))
default_scope = 'embodiedscan'
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
find_unused_parameters = True
launcher = 'pytorch'
load_from = None
log_level = 'INFO'
log_processor = dict(by_epoch=True, type='LogProcessor', window_size=50)
metainfo = dict(
    box_type_3d='euler-depth',
    classes=(
        'ceiling',
        'floor',
        'wall',
        'window',
        'chair',
        'bed',
        'sofa',
        'table',
        'tv',
        'furniture',
        'objects',
    ),
    occ_classes=(
        'ceiling',
        'floor',
        'wall',
        'window',
        'chair',
        'bed',
        'sofa',
        'table',
        'tv',
        'furniture',
        'objects',
    ))
model = dict(
    backbone=dict(
        depth=50,
        frozen_stages=1,
        init_cfg=dict(checkpoint='torchvision://resnet50', type='Pretrained'),
        norm_cfg=dict(requires_grad=False, type='BN'),
        norm_eval=True,
        num_stages=4,
        out_indices=(
            0,
            1,
            2,
            3,
        ),
        style='pytorch',
        type='mmdet.ResNet'),
    bbox_head=dict(
        in_channels=[
            128,
            128,
            128,
        ],
        num_classes=12,
        type='ImVoxelOccHead',
        use_semantic=True,
        volume_h=[
            20,
            10,
            5,
        ],
        volume_w=[
            20,
            10,
            5,
        ],
        volume_z=[
            8,
            4,
            2,
        ]),
    coord_type='DEPTH',
    data_preprocessor=dict(
        bgr_to_rgb=True,
        mean=[
            123.675,
            116.28,
            103.53,
        ],
        pad_size_divisor=32,
        std=[
            58.395,
            57.12,
            57.375,
        ],
        type='Det3DDataPreprocessor'),
    n_anchors=[
        40,
        40,
        32,
    ],
    n_voxels=[
        40,
        40,
        16,
    ],
    neck=dict(
        in_channels=[
            256,
            512,
            1024,
            2048,
        ],
        num_outs=4,
        out_channels=256,
        type='mmdet.FPN'),
    neck_3d=dict(
        in_channels=256,
        n_blocks=[
            1,
            1,
            1,
        ],
        out_channels=128,
        type='IndoorImVoxelNeck'),
    point_cloud_range=[
        -3.2,
        -3.2,
        -0.78,
        3.2,
        3.2,
        1.78,
    ],
    prior_generator=dict(
        ranges=[
            [
                -3.2,
                -3.2,
                -1.28,
                3.2,
                3.2,
                1.28,
            ],
        ],
        rotations=[
            0.0,
        ],
        type='AlignedAnchor3DRangeGenerator'),
    slicenet=dict(
        embed_dims=256,
        encoder=dict(
            num_layers=3,
            num_points_in_pillar=[
                4,
                4,
                4,
                4,
                4,
                4,
                4,
                4,
                4,
                4,
                4,
                4,
                4,
                4,
                4,
                4,
                4,
                4,
                4,
                4,
                4,
                4,
                4,
                4,
                4,
                4,
                4,
                4,
                4,
                4,
                4,
                4,
            ],
            pc_range=[
                -3.2,
                -3.2,
                -1.28,
                3.2,
                3.2,
                1.28,
            ],
            return_intermediate=False,
            slice_h=40,
            slice_num=[
                16,
                0,
                0,
            ],
            slice_w=40,
            slice_z=16,
            transformerlayers=dict(
                attn_cfgs=[
                    dict(
                        embed_dims=256,
                        num_levels=4,
                        num_slices=16,
                        tpv_h=40,
                        tpv_w=40,
                        tpv_z=16,
                        type='TPVCrossViewHybridAttention'),
                    dict(
                        deformable_attention=dict(
                            embed_dims=256,
                            floor_sampling_offset=False,
                            num_levels=4,
                            num_points=[
                                8,
                                8,
                                8,
                                8,
                                8,
                                8,
                                8,
                                8,
                                8,
                                8,
                                8,
                                8,
                                8,
                                8,
                                8,
                                8,
                                8,
                                8,
                                8,
                                8,
                                8,
                                8,
                                8,
                                8,
                                8,
                                8,
                                8,
                                8,
                                8,
                                8,
                                8,
                                8,
                            ],
                            num_slices=16,
                            num_z_anchors=[
                                4,
                                4,
                                4,
                                4,
                                4,
                                4,
                                4,
                                4,
                                4,
                                4,
                                4,
                                4,
                                4,
                                4,
                                4,
                                4,
                                4,
                                4,
                                4,
                                4,
                                4,
                                4,
                                4,
                                4,
                                4,
                                4,
                                4,
                                4,
                                4,
                                4,
                                4,
                                4,
                            ],
                            tpv_h=40,
                            tpv_w=40,
                            tpv_z=16,
                            type='TPVMSDeformableAttention3D'),
                        embed_dims=256,
                        num_cams=20,
                        num_slices=[
                            16,
                            0,
                            0,
                        ],
                        pc_range=[
                            -3.2,
                            -3.2,
                            -1.28,
                            3.2,
                            3.2,
                            1.28,
                        ],
                        tpv_h=40,
                        tpv_w=40,
                        tpv_z=16,
                        type='TPVImageCrossAttention'),
                ],
                feedforward_channels=512,
                ffn_dropout=0.1,
                operation_order=(
                    'self_attn',
                    'norm',
                    'cross_attn',
                    'norm',
                    'ffn',
                    'norm',
                ),
                type='TPVFormerLayer'),
            type='SliceOccEncoder'),
        num_cams=20,
        num_feature_levels=4,
        pc_range=[
            -3.2,
            -3.2,
            -1.28,
            3.2,
            3.2,
            1.28,
        ],
        positional_encoding=dict(
            col_num_embed=40,
            num_feats=128,
            row_num_embed=40,
            type='LearnedPositionalEncoding'),
        slice_h=40,
        slice_num=[
            16,
            0,
            0,
        ],
        slice_w=40,
        slice_z=32,
        type='SliceOccHead',
        use_semantic=True),
    type='DenseSliceOccPredictor',
    use_valid_mask=False,
    use_xyz_feat=True)
n_points = 100000
nbr_class = 12
num_points = [
    8,
    8,
    8,
    8,
    8,
    8,
    8,
    8,
    8,
    8,
    8,
    8,
    8,
    8,
    8,
    8,
    8,
    8,
    8,
    8,
    8,
    8,
    8,
    8,
    8,
    8,
    8,
    8,
    8,
    8,
    8,
    8,
]
num_points_h = []
num_points_in_pillar = [
    4,
    4,
    4,
    4,
    4,
    4,
    4,
    4,
    4,
    4,
    4,
    4,
    4,
    4,
    4,
    4,
    4,
    4,
    4,
    4,
    4,
    4,
    4,
    4,
    4,
    4,
    4,
    4,
    4,
    4,
    4,
    4,
]
num_points_in_pillar_h = []
num_points_in_pillar_w = []
num_points_in_pillar_z = [
    4,
    4,
    4,
    4,
    4,
    4,
    4,
    4,
    4,
    4,
    4,
    4,
    4,
    4,
    4,
    4,
    4,
    4,
    4,
    4,
    4,
    4,
    4,
    4,
    4,
    4,
    4,
    4,
    4,
    4,
    4,
    4,
]
num_points_w = []
num_points_z = [
    8,
    8,
    8,
    8,
    8,
    8,
    8,
    8,
    8,
    8,
    8,
    8,
    8,
    8,
    8,
    8,
    8,
    8,
    8,
    8,
    8,
    8,
    8,
    8,
    8,
    8,
    8,
    8,
    8,
    8,
    8,
    8,
]
num_slices_ = [
    16,
    0,
    0,
]
num_slices_h = 0
num_slices_w = 0
num_slices_z = 16
optim_wrapper = dict(
    clip_grad=dict(max_norm=35.0, norm_type=2),
    optimizer=dict(lr=0.0001, type='AdamW', weight_decay=0.01),
    type='OptimWrapper')
param_scheduler = dict(
    begin=0,
    by_epoch=True,
    end=32,
    gamma=0.1,
    milestones=[
        22,
        26,
    ],
    type='MultiStepLR')
point_cloud_range = [
    -3.2,
    -3.2,
    -0.78,
    3.2,
    3.2,
    1.78,
]
prior_generator = dict(
    ranges=[
        [
            -3.2,
            -3.2,
            -1.28,
            3.2,
            3.2,
            1.28,
        ],
    ],
    rotations=[
        0.0,
    ],
    type='AlignedAnchor3DRangeGenerator')
resume = False
scale_h = 1
scale_w = 1
scale_z = 1
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file='scannetpp_infos_val.pkl',
        box_type_3d='Euler-Depth',
        data_root='/data',
        filter_empty_gt=True,
        metainfo=dict(
            box_type_3d='euler-depth',
            classes=(
                'ceiling',
                'floor',
                'wall',
                'window',
                'chair',
                'bed',
                'sofa',
                'table',
                'tv',
                'furniture',
                'objects',
            ),
            occ_classes=(
                'ceiling',
                'floor',
                'wall',
                'window',
                'chair',
                'bed',
                'sofa',
                'table',
                'tv',
                'furniture',
                'objects',
            )),
        pipeline=[
            dict(
                type='LoadAnnotations3D',
                with_occupancy=True,
                with_visible_occupancy_masks=True),
            dict(
                n_images=20,
                transforms=[
                    dict(backend_args=None, type='LoadImageFromFile'),
                    dict(keep_ratio=False, scale=(
                        480,
                        480,
                    ), type='Resize'),
                ],
                type='MultiViewPipeline'),
            dict(type='ConstructMultiViewMasks'),
            dict(
                keys=[
                    'img',
                    'gt_bboxes_3d',
                    'gt_labels_3d',
                    'gt_occupancy',
                ],
                type='Pack3DDetInputs'),
        ],
        test_mode=True,
        type='ScannetppDataset'),
    drop_last=False,
    num_workers=1,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(type='OccupancyMetric')
test_pipeline = [
    dict(
        type='LoadAnnotations3D',
        with_occupancy=True,
        with_visible_occupancy_masks=True),
    dict(
        n_images=20,
        transforms=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(keep_ratio=False, scale=(
                480,
                480,
            ), type='Resize'),
        ],
        type='MultiViewPipeline'),
    dict(type='ConstructMultiViewMasks'),
    dict(
        keys=[
            'img',
            'gt_bboxes_3d',
            'gt_labels_3d',
            'gt_occupancy',
        ],
        type='Pack3DDetInputs'),
]
tpv_h_ = 40
tpv_w_ = 40
tpv_z_ = 16
train_cfg = dict(max_epochs=32, type='EpochBasedTrainLoop', val_interval=8)
train_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file='scannetpp_infos_train.pkl',
        box_type_3d='Euler-Depth',
        data_root='/data',
        filter_empty_gt=True,
        metainfo=dict(
            box_type_3d='euler-depth',
            classes=(
                'ceiling',
                'floor',
                'wall',
                'window',
                'chair',
                'bed',
                'sofa',
                'table',
                'tv',
                'furniture',
                'objects',
            ),
            occ_classes=(
                'ceiling',
                'floor',
                'wall',
                'window',
                'chair',
                'bed',
                'sofa',
                'table',
                'tv',
                'furniture',
                'objects',
            )),
        pipeline=[
            dict(
                type='LoadAnnotations3D',
                with_occupancy=True,
                with_visible_occupancy_masks=True),
            dict(
                n_images=20,
                transforms=[
                    dict(backend_args=None, type='LoadImageFromFile'),
                    dict(keep_ratio=False, scale=(
                        480,
                        480,
                    ), type='Resize'),
                ],
                type='MultiViewPipeline'),
            dict(type='ConstructMultiViewMasks'),
            dict(
                keys=[
                    'img',
                    'gt_bboxes_3d',
                    'gt_labels_3d',
                    'gt_occupancy',
                ],
                type='Pack3DDetInputs'),
        ],
        test_mode=False,
        type='ScannetppDataset'),
    num_workers=1,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))
train_pipeline = [
    dict(
        type='LoadAnnotations3D',
        with_occupancy=True,
        with_visible_occupancy_masks=True),
    dict(
        n_images=20,
        transforms=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(keep_ratio=False, scale=(
                480,
                480,
            ), type='Resize'),
        ],
        type='MultiViewPipeline'),
    dict(type='ConstructMultiViewMasks'),
    dict(
        keys=[
            'img',
            'gt_bboxes_3d',
            'gt_labels_3d',
            'gt_occupancy',
        ],
        type='Pack3DDetInputs'),
]
val_cfg = dict(type='ValLoop')
val_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file='scannetpp_infos_val.pkl',
        box_type_3d='Euler-Depth',
        data_root='/data',
        filter_empty_gt=True,
        metainfo=dict(
            box_type_3d='euler-depth',
            classes=(
                'ceiling',
                'floor',
                'wall',
                'window',
                'chair',
                'bed',
                'sofa',
                'table',
                'tv',
                'furniture',
                'objects',
            ),
            occ_classes=(
                'ceiling',
                'floor',
                'wall',
                'window',
                'chair',
                'bed',
                'sofa',
                'table',
                'tv',
                'furniture',
                'objects',
            )),
        pipeline=[
            dict(
                type='LoadAnnotations3D',
                with_occupancy=True,
                with_visible_occupancy_masks=True),
            dict(
                n_images=20,
                transforms=[
                    dict(backend_args=None, type='LoadImageFromFile'),
                    dict(keep_ratio=False, scale=(
                        480,
                        480,
                    ), type='Resize'),
                ],
                type='MultiViewPipeline'),
            dict(type='ConstructMultiViewMasks'),
            dict(
                keys=[
                    'img',
                    'gt_bboxes_3d',
                    'gt_labels_3d',
                    'gt_occupancy',
                ],
                type='Pack3DDetInputs'),
        ],
        test_mode=True,
        type='ScannetppDataset'),
    drop_last=False,
    num_workers=1,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = dict(type='OccupancyMetric')
visualizer = dict(
    save_dir='temp_dir',
    type='EmbodiedScanBaseVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
    ])
work_dir = 'work_dirs/sliceocc'
