
from .multi_scale_deformable_attn_function import MultiScaleDeformableAttnFunction_fp32
from mmcv.ops.multi_scale_deform_attn import multi_scale_deformable_attn_pytorch
import warnings
import torch
import torch.nn as nn
from mmengine.model import xavier_init, constant_init

import math
from mmengine.model import BaseModule
from embodiedscan.registry import MODELS

from mmcv.utils import ext_loader
ext_module = ext_loader.load_ext(
    '_ext', ['ms_deform_attn_backward', 'ms_deform_attn_forward'])


# forward twice cross_view att, change the order of queries

@MODELS.register_module()
class TPVCrossViewHybridAttention(BaseModule):
    
    def __init__(self, 
        tpv_h, tpv_w, tpv_z,
        num_slices=24,
        embed_dims=256, 
        num_heads=8, 
        num_points=4,
        num_anchors=2,
        init_mode=0,
        dropout=0.1,     
        **kwargs
    ):
        super().__init__()
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.num_levels = 4
        self.num_points = num_points
        self.num_anchors = num_anchors
        self.init_mode = init_mode
        self.num_slices = num_slices

        self.dropout = nn.ModuleList([
            nn.Dropout(dropout) for _ in range(self.num_slices)
        ])
        self.output_proj = nn.ModuleList([
            nn.Linear(embed_dims, embed_dims) for _ in range(self.num_slices)
        ])
        self.sampling_offsets = nn.ModuleList([
            nn.Linear(embed_dims, num_heads * self.num_levels * num_points * 2) for _ in range(self.num_slices)
        ])
        self.attention_weights = nn.ModuleList([
            nn.Linear(embed_dims, num_heads * self.num_levels * (num_points + 1)) for _ in range(self.num_slices)
        ])
        
        self.value_proj = nn.ModuleList([
            nn.Linear(embed_dims, embed_dims) for _ in range(self.num_slices)
        ])
        # should change the number from 3 to total plane numbers (query numbers)

        self.tpv_h, self.tpv_w, self.tpv_z = tpv_h, tpv_w, tpv_z
        self.init_weight()

    def init_weight(self):
        """Default initialization for Parameters of Module."""

        # self plane
        '''
        theta_self = torch.arange(
            self.num_heads, dtype=torch.float32) * (2.0 * math.pi / self.num_heads)
        grid_self = torch.stack([theta_self.cos(), theta_self.sin()], -1) # H, 2
        grid_self = grid_self.view(
            self.num_heads, 1, 2).repeat(1, self.num_points, 1)
        for j in range(self.num_points):
            grid_self[:, j, :] *= (j + 1) / 2
        
        if self.init_mode == 0:
            # num_phi = 4
            phi = torch.arange(4, dtype=torch.float32) * (2.0 * math.pi / 4) # 4
            assert self.num_heads % 4 == 0
            num_theta = int(self.num_heads / 4)
            theta = torch.arange(num_theta, dtype=torch.float32) * (math.pi / num_theta) + (math.pi / num_theta / 2) # 3
            x = torch.matmul(theta.sin().unsqueeze(-1), phi.cos().unsqueeze(0)).flatten()
            y = torch.matmul(theta.sin().unsqueeze(-1), phi.sin().unsqueeze(0)).flatten()
            z = theta.cos().unsqueeze(-1).repeat(1, 4).flatten()
            xyz = torch.stack([x, y, z], dim=-1) # H, 3

        elif self.init_mode == 1:        
            xyz = [
                [0, 0, 1],
                [0, 0, -1],
                [0, 1, 0],
                [0, -1, 0],
                [1, 0, 0],
                [-1, 0, 0]
            ]
            xyz = torch.tensor(xyz, dtype=torch.float32)

        grid_hw = xyz[:, [0, 1]] # H, 2
        grid_zh = xyz[:, [2, 0]]
        grid_wz = xyz[:, [1, 2]]


        for i in range(self.num_slices):
            grid = torch.stack([grid_hw, grid_zh, grid_wz], dim=1) # H, 3, 2
            grid = grid.unsqueeze(2).repeat(1, 1, self.num_points, 1)
            
            grid = grid.reshape(self.num_heads, self.num_levels, self.num_anchors, -1, 2)
            for j in range(self.num_points // self.num_anchors):
                grid[:, :, :, j, :] *= 2 * (j + 1)
            grid = grid.flatten(2, 3)
            grid[:, i, :, :] = grid_self
            
            constant_init(self.sampling_offsets[i], 0.)
            self.sampling_offsets[i].bias.data = grid.view(-1)
        '''
        for i in range(self.num_slices): # num planes
            constant_init(self.sampling_offsets[i], 0.)
            thetas = torch.arange(
                self.num_heads,
                dtype=torch.float32) * (2.0 * math.pi / self.num_heads)
            grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
            grid_init = (grid_init /
                        grid_init.abs().max(-1, keepdim=True)[0]).view(
                self.num_heads, 1, 1,
                2).repeat(1, self.num_levels, self.num_points, 1)
            #print(grid_init.shape)
            grid_init = grid_init.reshape(self.num_heads, self.num_levels, self.num_anchors, -1, 2)
            for j in range(self.num_points // self.num_anchors):
                grid_init[:, :, :, j, :] *= j + 1

            constant_init(self.attention_weights[i], val=0., bias=0.)
            #attn_bias = torch.zeros(self.num_heads, self.num_slices, self.num_points + 1)
            #attn_bias[:, i, -1] = 10
            #self.attention_weights[i].bias.data = attn_bias.flatten()
            constant_init(self.attention_weights[i], val=0., bias=0.)
            xavier_init(self.value_proj[i], distribution='uniform', bias=0.)
            xavier_init(self.output_proj[i], distribution='uniform', bias=0.)    
        
    def get_sampling_offsets_and_attention(self, queries):
        offsets = []
        attns = []
        for i, (query, fc, attn) in enumerate(zip(queries, self.sampling_offsets, self.attention_weights)):
            bs, l, d = query.shape
            offset = fc.to(query.device)(query).reshape(bs, l, self.num_heads, self.num_levels, self.num_points, -1, 2)
            offsets.append(offset)
             #torch.Size([1, 400, 256])
            #print(query.shape) torch.Size([1, 400, 256])
            attention = attn(query).reshape(bs, l, self.num_heads, self.num_levels, -1)
            #print(attention.shape) torch.Size([1, 400, 8, 4, 5])
            level_attention = attention[:, :, :, :, -1:].softmax(-2) # bs, l, H, 3, 1
            attention = attention[:, :, :, :, :-1]
            attention = attention.softmax(-1) # bs, l, H, 3, p
            attention = attention * level_attention
            attns.append(attention)
            '''
            attention = attn.to(query.device)(query)
            #print(attention.shape)
            attention = attention.reshape(bs, l, self.num_heads, self.num_levels, -1)
            #level_attention = attention[:, :, :, :, -1:].softmax(-2) # bs, l, H, 3, 1
            #attention = attention[:, :, :, :, :-1]
            attention = attention.softmax(-1) # bs, l, H, 3, p
            attention = attention.view(bs, l, self.num_heads, self.num_levels, 1, -1)
            attention = attention.permute(0, 1, 4, 2, 3, 5).flatten(1, 2) 
            #attention = attention * level_attention
            attns.append(attention)
            '''
        
        offsets = torch.cat(offsets, dim=1)
        attns = torch.cat(attns, dim=1)
    
        return offsets, attns

    def reshape_output(self, output, lens):
        bs, _, d = output.shape
        split_sizes = [lens[i] for i in range(self.num_slices)]
        outputs = torch.split(output, split_sizes, dim=1)        
        #outputs = torch.split(output, [lens[0], lens[1], lens[2]], dim=1)
        return outputs

    def forward(self,                
                query,
                value,
                identity=None,
                query_pos=None,
                reference_points=None,
                spatial_shapes=None,
                level_start_index=None,
                **kwargs):
        identity = query if identity is None else identity
        if query_pos is not None:
            query = [q + p for q, p in zip(query, query_pos)]

        # value proj
        #original code for only hw planes
        #print(len(query)) 24
        query_lens = [q.shape[1] for q in query]
        value = [layer(v) for layer, v in zip(self.value_proj, value)]
        value = torch.cat(value, dim=1)
        bs, num_value, _ = value.shape
        value = value.view(bs, num_value, self.num_heads, -1)
        #print(len(query))
        # sampling offsets and weights
        sampling_offsets, attention_weights = self.get_sampling_offsets_and_attention(query)

        if reference_points.shape[-1] == 2:
            """
            For each tpv query, it owns `num_Z_anchors` in 3D space that having different heights.
            After proejcting, each tpv query has `num_Z_anchors` reference points in each 2D image.
            For each referent point, we sample `num_points` sampling points.
            For `num_Z_anchors` reference points,  it has overall `num_points * num_Z_anchors` sampling points.
            """
            offset_normalizer = torch.stack(
                [spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)
            #print(reference_points.shape)
            '''
            bs, num_query, _, num_Z_anchors, xy = reference_points.shape
            reference_points = reference_points[:, :, None, :, :, None, :]
            #print(sampling_offsets.shape, offset_normalizer.shape) #torch.Size([1, 5760, 8, 4, 4, 1, 2]) torch.Size([24, 2])
            sampling_offsets = sampling_offsets.squeeze(-2) / \
                offset_normalizer[None, None, None, :, None, :]
            bs, num_query, num_heads, num_levels, num_all_points, xy = sampling_offsets.shape
            sampling_offsets = sampling_offsets.view(
                bs, num_query, num_heads, num_levels, num_Z_anchors, num_all_points // num_Z_anchors, xy)
            sampling_locations = reference_points + sampling_offsets
            bs, num_query, num_heads, num_levels, num_points, num_Z_anchors, xy = sampling_locations.shape
            assert num_all_points == num_points * num_Z_anchors
            '''

            sampling_offsets = sampling_offsets.squeeze(-2)

            if reference_points.shape[-1] == 2:
                #print(reference_points.shape, sampling_offsets.shape, offset_normalizer.shape)
                offset_normalizer = torch.stack(
                    [spatial_shapes[..., 1], spatial_shapes[..., 0]], -1).to(query[0].device)
                sampling_locations = reference_points[:, :, None, :, None, :] \
                    + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
                #print("cross", torch.max(sampling_locations))
            #sampling_locations = sampling_locations.view(
            #    bs, num_query, num_heads, num_levels, num_all_points, xy)
        else:
            raise ValueError(
                f'Last dim of reference_points must be'
                f' 2, but get {reference_points.shape[-1]} instead.')
        
        if torch.cuda.is_available():
            if value.dtype == torch.float16:
                MultiScaleDeformableAttnFunction = MultiScaleDeformableAttnFunction_fp32
            else:
                MultiScaleDeformableAttnFunction = MultiScaleDeformableAttnFunction_fp32
            output = MultiScaleDeformableAttnFunction.apply(
                value, spatial_shapes, level_start_index, sampling_locations,
                attention_weights, 64)
        else:
            output = multi_scale_deformable_attn_pytorch(
                value, spatial_shapes, sampling_locations, attention_weights)

        outputs = self.reshape_output(output, query_lens)

        results = []
        for out, layer, drop, residual in zip(outputs, self.output_proj, self.dropout, identity):
            results.append(residual + drop(layer(out)))

        return results