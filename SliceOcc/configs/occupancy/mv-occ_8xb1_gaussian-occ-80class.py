_base_ = ['../default_runtime.py']
n_points = 100000

# origin for multi-view scannet is set to 0.5
# -1.28~1.28 -> -0.78~1.78
point_cloud_range = [-3.2, -3.2, -0.78, 3.2, 3.2, 1.78]
cam_point_range = [-3.2, -3.2, -1.28, 3.2, 3.2, 1.28]



prior_generator = dict(type='AlignedAnchor3DRangeGenerator',
                       ranges=[[-3.2, -3.2, -1.28, 3.2, 3.2, 1.28]],
                       rotations=[.0])


_dim_ = 256
_pos_dim_ = _dim_//2
_ffn_dim_ = _dim_*2
_num_levels_ = 4
_num_cams_ = 20
tpv_h_ = 40
tpv_w_ = 40
tpv_z_ = 16
anchor_z_ = 16
num_slices_ = 8
scale_h = 1
scale_w = 1
scale_z = 1

nbr_class = 81

embed_dims = 128
num_groups = 4
num_levels = 4
num_decoder = 4
num_single_frame_decoder = 1
pc_range = [-3.2, -3.2, -0.78, 3.2, 3.2, 1.78]
scale_range = [0.08, 0.64]
xyz_coordinate = 'cartesian'
phi_activation = 'sigmoid'
use_deformable_func = True  
include_opa = True
#load_from = 'ckpts/r101_dcn_fcos3d_pretrain.pth'
semantics = True
semantic_dim = 81


model = dict(
    type='GaussianOccPredictor',
    use_valid_mask=False,
    use_xyz_feat=True,
    point_cloud_range=point_cloud_range,
    data_preprocessor=dict(type='Det3DDataPreprocessor',
                           mean=[123.675, 116.28, 103.53],
                           std=[58.395, 57.12, 57.375],
                           bgr_to_rgb=True,
                           pad_size_divisor=32),
    backbone=dict(type='mmdet.ResNet',
                  depth=50,
                  num_stages=4,
                  out_indices=(0, 1, 2, 3),
                  frozen_stages=1,
                  norm_cfg=dict(type='BN', requires_grad=False),
                  norm_eval=True,
                  init_cfg=dict(type='Pretrained',
                                checkpoint='torchvision://resnet50'),
                  style='pytorch'),
    neck=dict(type='mmdet.FPN',
              in_channels=[256, 512, 1024, 2048],
              out_channels=256,
              num_outs=4),
    lifter=dict(
        type='GaussianLifter',
        num_anchor=25600,
        embed_dims=embed_dims,
        anchor_grad=True,
        feat_grad=False,
        phi_activation=phi_activation,
        semantics=semantics,
        semantic_dim=semantic_dim,
        include_opa=include_opa,
    ),
    encoder=dict(
        type='GaussianOccEncoder',
        anchor_encoder=dict(
            type='SparseGaussian3DEncoder',
            embed_dims=embed_dims, 
            include_opa=include_opa,
            semantics=semantics,
            semantic_dim=semantic_dim
        ),
        norm_layer=dict(type="LN", normalized_shape=embed_dims),
        ffn=dict(
            type="AsymmetricFFN",
            in_channels=embed_dims * 2,
            embed_dims=embed_dims,
            feedforward_channels=embed_dims * 4,
        ),
        deformable_model=dict(
            type='DeformableFeatureAggregation',
            embed_dims=embed_dims,
            num_groups=num_groups,
            num_levels=num_levels,
            num_cams=6,
            attn_drop=0.15,
            use_deformable_func=use_deformable_func,
            use_camera_embed=True,
            residual_mode="cat",
            kps_generator=dict(
                type="SparseGaussian3DKeyPointsGenerator",
                embed_dims=embed_dims,
                phi_activation=phi_activation,
                xyz_coordinate=xyz_coordinate,
                num_learnable_pts=6,
                fix_scale=[
                    [0, 0, 0],
                    [0.45, 0, 0],
                    [-0.45, 0, 0],
                    [0, 0.45, 0],
                    [0, -0.45, 0],
                    [0, 0, 0.45],
                    [0, 0, -0.45],
                ],
                pc_range=pc_range,
                scale_range=scale_range
            ),
        ),
        refine_layer=dict(
            type='SparseGaussian3DRefinementModule',
            embed_dims=embed_dims,
            pc_range=pc_range,
            scale_range=scale_range,
            restrict_xyz=True,
            unit_xyz=[4.0, 4.0, 1.0],
            refine_manual=[0, 1, 2],
            phi_activation=phi_activation,
            semantics=semantics,
            semantic_dim=semantic_dim,
            include_opa=include_opa,
            xyz_coordinate=xyz_coordinate,
            semantics_activation='softplus',
        ),
        spconv_layer=dict(
            _delete_=True,
            type="SparseConv3D",
            in_channels=embed_dims,
            embed_channels=embed_dims,
            pc_range=pc_range,
            grid_size=[0.5, 0.5, 0.5],
            phi_activation=phi_activation,
            xyz_coordinate=xyz_coordinate,
            use_out_proj=True,
        ),
        num_decoder=num_decoder,
        num_single_frame_decoder=num_single_frame_decoder,
        operation_order=[
            "deformable",
            "ffn",
            "norm",
            "refine",
        ] * num_single_frame_decoder + [
            "spconv",
            "norm",
            "deformable",
            "ffn",
            "norm",
            "refine",
        ] * (num_decoder - num_single_frame_decoder),
    ),
    head=dict(
        type='GaussianHead',
        apply_loss_type='random_1',
        num_classes=semantic_dim + 1,
        empty_args=dict(
            _delete_=True,
            mean=[0, 0, -1.0],
            scale=[100, 100, 8.0],
        ),
        with_empty=True,
        cuda_kwargs=dict(
            _delete_=True,
            scale_multiplier=3,
            H=40, W=40, D=16,
            pc_min=[-3.2, -3.2, -0.78],
            grid_size=0.5),
    ),
    neck_3d=dict(type='IndoorImVoxelNeck',
                 in_channels=256,
                 out_channels=128,
                 n_blocks=[1, 1, 1]),
    bbox_head=dict(
        type='ImVoxelOccHead',
        volume_h=[20, 10, 5],
        volume_w=[20, 10, 5],
        volume_z=[8, 4, 2],
        num_classes=81,  # TO Be changed
        in_channels=[128, 128, 128], #[128, 128, 128],
        use_semantic=True),
    prior_generator=prior_generator,
    n_voxels=[40, 40, 16],  
    n_anchors=[40, 40, 16],
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
    dict(
        type='MultiViewPipeline',
        n_images=_num_cams_,
        transforms=[
            dict(type='LoadImageFromFile', backend_args=backend_args),
            dict(type='Resize', scale=(480, 480), keep_ratio=False)
        ]),
    dict(type='ConstructMultiViewMasks'),
    dict(type='Pack3DDetInputs', keys=['img', 'gt_bboxes_3d', 'gt_labels_3d', 'gt_occupancy'])
]

test_pipeline = [
    dict(type='LoadAnnotations3D',
         with_occupancy=True,
         with_visible_occupancy_masks=True),
    dict(
        type='MultiViewPipeline',
        n_images=_num_cams_,
        transforms=[
            dict(type='LoadImageFromFile', backend_args=backend_args),
            dict(type='Resize', scale=(480, 480), keep_ratio=False)
        ]),
    dict(type='ConstructMultiViewMasks'),
    dict(type='Pack3DDetInputs', keys=['img', 'gt_bboxes_3d', 'gt_labels_3d', 'gt_occupancy'])
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

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=24, val_interval=24) #32
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
                       milestones=[16, 20],
                       gamma=0.1)

custom_hooks = [dict(type='EmptyCacheHook', after_iter=True)]

# hooks
default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=24, max_keep_ckpts=24))

# runtime
find_unused_parameters = True  # only 1 of 4 FPN outputs is used
visualizer = dict(type='EmbodiedScanBaseVisualizer', vis_backends=[dict(type='LocalVisBackend')], save_dir='temp_dir')