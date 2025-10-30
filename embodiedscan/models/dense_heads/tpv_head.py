import torch
import torch.nn as nn
from torch.nn.init import normal_
import numpy as np
import math

from mmengine.model import BaseModule
from mmseg.models import HEADS
from mmcv.cnn.bricks.transformer import build_positional_encoding, \
    build_transformer_layer_sequence
from mmcv.cnn import build_conv_layer

#from .modules.cross_view_hybrid_attention import sliceCrossViewHybridAttention
#from .modules.image_cross_attention import sliceMSDeformableAttention3D
from .modules.sliding_window_attention import SlidingWindowAttention
from embodiedscan.registry import MODELS
from embodiedscan.structures.points.base_points import TwoStageKMeans

from embodiedscan.models.losses.occ_loss import (geo_scal_loss,
                                                 occ_multiscale_supervision,
                                                 sem_scal_loss)
                               
from embodiedscan.utils.typing_config import SampleList

@MODELS.register_module()
class SliceOccHead(BaseModule):

    def __init__(self,
                 positional_encoding=None,
                 slice_h=30,
                 slice_w=30,
                 slice_z=30,
                 slice_num=4,
                 pc_range=[-51.2, -51.2, -5, 51.2, 51.2, 3],
                 num_feature_levels=4,
                 num_cams=6,
                 encoder=None,
                 embed_dims=256,
                 **kwargs):
        super().__init__()

        self.slice_h = slice_h
        self.slice_w = slice_w
        self.slice_z = slice_z
        self.pc_range = pc_range
        self.embed_dims = embed_dims
        self.num_feature_levels = num_feature_levels
        self.num_cams = num_cams
        self.fp16_enabled = False
        self.slice_num = slice_num
        self.slice_embedding = nn.ModuleList([nn.Embedding(self.slice_h * self.slice_w, self.embed_dims) for i in range(self.slice_num[0])])

        self.encoder = build_transformer_layer_sequence(encoder)
        self.level_embeds = nn.Parameter(
            torch.Tensor(self.num_feature_levels, self.embed_dims))
        self.cams_embeds = nn.Parameter(
            torch.Tensor(self.num_cams, self.embed_dims))


    def init_weights(self):
        """Initialize the transformer weights."""
        
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        
        # print("Hello modules!!!!!!!!!!!!!!!!!!!")
        # for m in self.modules():
        #     print(type(m))
         
        # for m in self.modules():
        #     if isinstance(m, sliceMSDeformableAttention3D) or isinstance(m, sliceCrossViewHybridAttention) or isinstance(m, SlidingWindowAttention):
        #         try:
        #             m.init_weight()
        #         except AttributeError:
        #             m.init_weights()
        normal_(self.level_embeds)
        normal_(self.cams_embeds)


    def forward(self, mlvl_feats, img_metas, img_volume_feats):
        """Forward function.
        Args:
            mlvl_feats (tuple[Tensor]): Features from the upstream
                network, each is a 5D-tensor with shape
                (B, N, C, H, W).
        """
        bs = mlvl_feats[0].shape[0]
        dtype = mlvl_feats[0].dtype
        device = mlvl_feats[0].device

        # slice queries and height embeds
 
        split_size = self.slice_z // self.slice_num[0]
        sliced_volumes = torch.split(img_volume_feats, split_size_or_sections=split_size, dim=4)
        reshaped_queries = []
        voxel_heights = []
        
        for s_idx, q in enumerate(sliced_volumes): # (1,C,40,40,4)
            q_f = q[:,:,:,:,0]            
            b, c, h, w = q_f.size() 
            q_f = q_f.contiguous().view(b, c, h*w).permute(0,2,1) + self.slice_embedding[s_idx].weight.to(q_f.dtype)
            q_c = q[:,:,:,:,-1]
            q_c = q_c.contiguous().view(b, c, h*w).permute(0,2,1) + self.slice_embedding[s_idx].weight.to(q_c.dtype)         
            reshaped_queries.append(q_f)
            reshaped_queries.append(q_c)

            voxel_heights.append(round(q.size(-1) / 2))
        

        voxel_heights = adjust_sequence_to_target(voxel_heights, self.slice_z)
        assert sum(voxel_heights) == self.slice_z

        feat_flatten = []
        spatial_shapes = []
        for lvl, feat in enumerate(mlvl_feats):      
            bs, num_cam, c, h, w = feat.shape
            spatial_shape = (h, w)
            feat = feat.flatten(3).permute(1, 0, 3, 2) # num_cam, bs, hw, c
            feat = feat + self.cams_embeds[:, None, None, :].to(dtype)
            feat = feat + self.level_embeds[None, None, lvl:lvl+1, :].to(dtype)
            spatial_shapes.append(spatial_shape)
            feat_flatten.append(feat)

        feat_flatten = torch.cat(feat_flatten, 2) # num_cam, bs, hw++, c
        spatial_shapes = torch.as_tensor(
            spatial_shapes, dtype=torch.long, device=device)
        level_start_index = torch.cat((spatial_shapes.new_zeros(
            (1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
        feat_flatten = feat_flatten.permute(0, 2, 1, 3)  # (num_cam, H*W, bs, embed_dims)

        slice_embed, ref_3ds = self.encoder(
            reshaped_queries, 
            feat_flatten,
            feat_flatten,
            slice_order=(range(self.slice_num[0]), range(self.slice_num[1]), range(self.slice_num[2])),
            slice_h=self.slice_h,
            slice_w=self.slice_w,
            slice_z=self.slice_z,
            slice_pos=None,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            img_metas=img_metas,
        )

        #slice_z = 16 // self.slice_num[0]
        slice_z = (self.slice_z // 2) // self.slice_num[0]
        occ_slices_z = []              
        voxel_heights = [split_size] * self.slice_num[0]

        '''
        import os
        scene_id = img_metas[0]['scan_id']
        scene_id = scene_id.replace("/", "_")
        save_dir = "/mnt/data/ljn/code/EmbodiedScan/vis/vis_feat/" 
        for e_idx, emb in enumerate(slice_embed):
            emb = emb.cpu().detach().permute(0, 2, 1)
            if emb.size(-1)==1600:
                emb = emb.reshape(1,256,40,40)
            elif emb.size(-1)==640:
                emb = emb.reshape(1,256,40,16)
            save_dir_ = os.path.join(save_dir, scene_id)
            if not os.path.exists(save_dir_):
                os.makedirs(save_dir_)
            save_dir_ = save_dir_ + "/{}.pth".format(e_idx)
            torch.save(emb, save_dir_)
        '''
        slice_embed_z = slice_embed[0:self.slice_num[0]*2]

        for i in range(0, len(slice_embed_z), 2):
            slice_floor = slice_embed_z[i]
            slice_celing = slice_embed_z[i+1]
            bs, _, c = slice_floor.shape
            slice_floor = slice_floor.permute(0, 2, 1).reshape(bs, c, self.slice_h, self.slice_w)
            slice_celing = slice_celing.permute(0, 2, 1).reshape(bs, c, self.slice_h, self.slice_w)
            stacked_features = torch.stack([slice_floor, slice_celing], dim=-1)
            interpolated_features = torch.nn.functional.interpolate(stacked_features, size=(self.slice_h, self.slice_w, slice_z), mode='trilinear', align_corners=False)
            occ_slices_z.append(interpolated_features)

        slice_occ_z = torch.cat(occ_slices_z, dim=4)
        slice_occ = slice_occ_z #+ img_volume_feats

        return slice_occ


def adjust_sequence_to_target(sequence, target_sum):
    current_sum = sum(sequence)
    difference = current_sum - target_sum
    
    if difference == 0:
        return sequence
    
    adjusted_sequence = sequence[:]
  
    while difference > 0:
  
        max_value = max(adjusted_sequence)
        max_index = adjusted_sequence.index(max_value)
        
        adjusted_sequence[max_index] -= 1
        difference -= 1
 
    while difference < 0:

        min_value = min(adjusted_sequence)
        min_index = adjusted_sequence.index(min_value)
        
        adjusted_sequence[min_index] += 1
        difference += 1
    
    return adjusted_sequence