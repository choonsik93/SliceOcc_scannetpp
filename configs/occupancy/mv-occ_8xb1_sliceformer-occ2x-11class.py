_base_ = ['../default_runtime.py']
n_points = 100000



# origin for multi-view scannet is set to 0.5
# -1.28~1.28 -> -0.78~1.78
point_cloud_range = [-6.0, -6.0, -0.78, 6.0, 6.0, 3.22]
cam_point_range = [-6.0, -6.0, -0.78, 6.0, 6.0, 3.22]
# voxel_dims = [240, 240, 80]
# voxel_size = 0.05

prior_generator = dict(type='AlignedAnchor3DRangeGenerator',
                       ranges=[[-6.0, -6.0, -0.78, 6.0, 6.0, 3.22]],
                       rotations=[.0])

out_h = 240
out_w = 240
out_z = 80
_dim_ = 256
_pos_dim_ = _dim_//2
_ffn_dim_ = _dim_*2
_num_levels_ = 4
_num_cams_ = 1
tpv_h_ = 60
tpv_w_ = 60
tpv_z_ = 20
anchor_z_ = 40 #16
num_slices_z = 20
num_slices_w = 0
num_slices_h = 0
num_slices_ = [num_slices_z, num_slices_w, num_slices_h]
scale_h = 1
scale_w = 1
scale_z = 1
num_points_in_pillar_z = [4] * 2 * num_slices_z #这里也可以做消融实验，对显存影响较大
num_points_z = [8] * 2 * num_slices_z
num_points_in_pillar_w = [3] * 2 * num_slices_w #这里也可以做消融实验，对显存影响较大
num_points_w = [6] * 2 * num_slices_w
num_points_in_pillar_h = [3] * 2 * num_slices_h #这里也可以做消融实验，对显存影响较大
num_points_h = [6] * 2 * num_slices_h
num_points_in_pillar = num_points_in_pillar_z #+ num_points_in_pillar_w + num_points_in_pillar_h
num_points = num_points_z #+ num_points_w + num_points_h
nbr_class = 12



model = dict(
    type='DenseSliceOccPredictor',
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
    slicenet=dict(
        type='SliceOccHead',
        slice_h=tpv_h_,
        slice_w=tpv_w_,
        slice_z=anchor_z_,
        slice_num=num_slices_,
        pc_range=cam_point_range,
        num_feature_levels=_num_levels_,
        num_cams=_num_cams_,
        embed_dims=_dim_,
        use_semantic=True,
        positional_encoding=dict(
            type='LearnedPositionalEncoding',
            num_feats=_pos_dim_,
            row_num_embed=tpv_h_,
            col_num_embed=tpv_w_),
        encoder=dict(
            type='SliceOccEncoder',
            slice_h=tpv_h_,
            slice_w=tpv_w_,
            slice_z=tpv_z_,
            slice_num=num_slices_,
            num_layers=3,
            pc_range=cam_point_range,
            num_points_in_pillar=num_points_in_pillar_z,
            return_intermediate=False,
            transformerlayers=dict(
                type='TPVFormerLayer',
                attn_cfgs=[
                    dict(
                        type='TPVCrossViewHybridAttention',
                        tpv_h=tpv_h_,
                        tpv_w=tpv_w_,
                        tpv_z=tpv_z_,
                        num_slices=num_slices_[0]+num_slices_[1]+num_slices_[2],
                        embed_dims=_dim_,
                        num_levels=4),                    
                    dict(
                        type='TPVImageCrossAttention',
                        pc_range=cam_point_range,
                        num_cams=_num_cams_,
                        num_slices=[num_slices_[0], num_slices_[1], num_slices_[2]],
                        deformable_attention=dict(
                            type='TPVMSDeformableAttention3D',
                            embed_dims=_dim_,
                            num_points=num_points,
                            num_z_anchors=num_points_in_pillar,
                            num_slices=num_slices_[0]+num_slices_[1]+num_slices_[2],
                            num_levels=_num_levels_,
                            floor_sampling_offset=False,
                            tpv_h=tpv_h_,
                            tpv_w=tpv_w_,
                            tpv_z=tpv_z_,
                        ),
                        embed_dims=_dim_,
                        tpv_h=tpv_h_,
                        tpv_w=tpv_w_,
                        tpv_z=tpv_z_,
                    ),                     
                ],
                feedforward_channels=_ffn_dim_,
                ffn_dropout=0.1,
                operation_order=('self_attn', 'norm', 'cross_attn', 'norm', 
                                 'ffn', 'norm')))),
            
    neck_3d=dict(type='IndoorImVoxelNeck',
                 in_channels=256,
                 out_channels=128,
                 n_blocks=[1, 1, 1]),
    bbox_head=dict(
        type='ImVoxelOccHead',
        # volume_h=[out_w, out_w // 2, out_w // 4],
        # volume_w=[out_h, out_h // 2, out_h // 4],
        # volume_z=[out_z, out_z // 2, out_z // 4],
        volume_h=out_w,
        volume_w=out_h,
        volume_z=out_z,
        num_classes=12,  # TO Be changed
        in_channels=[128, 128, 128], #[128, 128, 128],
        use_semantic=True),
    prior_generator=prior_generator,
    n_voxels=[out_w, out_h, out_z],  
    n_anchors= [tpv_h_, tpv_w_, anchor_z_], #[40, 40, 16], #32
    coord_type='DEPTH',
)

dataset_type = 'Scannetpp2xDataset'
data_root = '/data'
class_names = ('ceiling', 'floor', 'wall', 'window', 'chair', 'bed', 'sofa',
               'table', 'tv', 'furniture', 'objects')

metainfo = dict(classes=class_names,
                occ_classes=class_names,
                box_type_3d='euler-depth')
backend_args = None

train_pipeline = [
    dict(type='LoadAnnotations3D',
         with_occupancy=True,
         with_visible_occupancy_masks=False),
    dict(
        type='MultiViewPipeline',
        n_images=_num_cams_,
        transforms=[
            dict(type='LoadImageFromFile', backend_args=backend_args),
            dict(type='Resize', scale=(480, 480), keep_ratio=False)
        ]),
    dict(type='Pack3DDetInputs', keys=['img', 'gt_bboxes_3d', 'gt_labels_3d', 'gt_occupancy'])
]

test_pipeline = [
    dict(type='LoadAnnotations3D',
         with_occupancy=True,
         with_visible_occupancy_masks=False),
    dict(
        type='MultiViewPipeline',
        n_images=_num_cams_,
        transforms=[
            dict(type='LoadImageFromFile', backend_args=backend_args),
            dict(type='Resize', scale=(480, 480), keep_ratio=False)
        ]),
    dict(type='Pack3DDetInputs', keys=['img', 'gt_bboxes_3d', 'gt_labels_3d', 'gt_occupancy'])
]

train_dataloader = dict(batch_size=1,
                        num_workers=1,
                        persistent_workers=True,
                        sampler=dict(type='DefaultSampler', shuffle=True),
                        dataset=dict(type=dataset_type,
                                     data_root=data_root,
                                     ann_file='scannetpp_infos_2x_train.pkl',
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
                                   ann_file='scannetpp_infos_2x_val.pkl',
                                   pipeline=test_pipeline,
                                   test_mode=True,
                                   filter_empty_gt=True,
                                   box_type_3d='Euler-Depth',
                                   metainfo=metainfo))
test_dataloader = val_dataloader

val_evaluator = dict(type='OccupancyMetric')
test_evaluator = val_evaluator

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=32, val_interval=8) #32
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# optimizer
optim_wrapper = dict(type='OptimWrapper',
                     optimizer=dict(type='AdamW', lr=1e-4, weight_decay=0.01),
                     clip_grad=dict(max_norm=35., norm_type=2))
param_scheduler = dict(type='MultiStepLR',
                       begin=0,
                       end=32,
                       by_epoch=True,
                       milestones=[22, 26],
                       gamma=0.1)

custom_hooks = [dict(type='EmptyCacheHook', after_iter=True)]

# hooks
default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=32, max_keep_ckpts=32),)

# runtime
find_unused_parameters = True  # only 1 of 4 FPN outputs is used
visualizer = dict(type='EmbodiedScanBaseVisualizer', vis_backends=[dict(type='LocalVisBackend')], save_dir='temp_dir')
