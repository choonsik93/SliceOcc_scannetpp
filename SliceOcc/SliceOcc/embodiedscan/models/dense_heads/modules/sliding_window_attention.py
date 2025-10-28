
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
class SlidingWindowAttention(BaseModule):
    
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
        #self.slide_embedding = nn.Embedding(5 * 5 * 3, self.embed_dims)
        self.height_embedding = nn.Parameter(
            torch.Tensor(3, self.embed_dims), requires_grad=True) #all slice-level

        self.dropout = nn.Dropout(dropout)
        self.output_proj = nn.Linear(embed_dims, embed_dims)     
        self.sampling_offsets = nn.Linear(embed_dims, num_heads * self.num_levels * num_points * 2)       
        self.attention_weights = nn.Linear(embed_dims, num_heads * self.num_levels * (num_points + 1))  
        self.value_proj = nn.Linear(embed_dims, embed_dims)    
        # should change the number from 3 to total plane numbers (query numbers)

        self.tpv_h, self.tpv_w, self.tpv_z = tpv_h, tpv_w, tpv_z
        self.init_weight()

    @staticmethod
    def get_reference_points(H, W, dim='2d', bs=1, device='cuda', dtype=torch.float):
        """Get the reference points used in SCA and TSA.
        Args:
            H, W: spatial shape of tpv.
            Z: hight of pillar.
            D: sample D points uniformly from each pillar.
            device (obj:`device`): The device where
                reference_points should be.
        Returns:
            Tensor: reference points used in decoder, has \
                shape (bs, num_keys, num_levels, 2).
        """

        # reference points in 3D space, used in spatial cross-attention (SCA)


        if dim == '2d':
            ref_2d_list = []
            Z = 3
            for i in range(3): # height number
                #for i, z_idx in enumerate(slice_order):
                ref_y, ref_x = torch.meshgrid(
                        torch.linspace(
                                0.5, H - 0.5, H, dtype=dtype, device=device),
                        torch.linspace(
                                0.5, W - 0.5, W, dtype=dtype, device=device)
                        )
                ref_y = ref_y.reshape(-1)[None] / H
                ref_x = ref_x.reshape(-1)[None] / W
   
                ref_2d = torch.stack((ref_x, ref_y), -1)
                ref_2d = ref_2d.repeat(bs, 1, 1).unsqueeze(2)
                ref_2d_list.append(ref_2d)
            
            return torch.cat(ref_2d_list, dim=1)


    def init_weight(self):
        """Default initialization for Parameters of Module."""

        constant_init(self.sampling_offsets, 0.)
        thetas = torch.arange(
                self.num_heads,
                dtype=torch.float32) * (2.0 * math.pi / self.num_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (grid_init /
                        grid_init.abs().max(-1, keepdim=True)[0]).view(
                self.num_heads, 1, 1,
                2).repeat(1, self.num_levels, self.num_points, 1)

        grid_init = grid_init.reshape(self.num_heads, self.num_levels, self.num_anchors, -1, 2)
        for j in range(self.num_points // self.num_anchors):
            grid_init[:, :, :, j, :] *= j + 1

        constant_init(self.attention_weights, val=0., bias=0.)
        xavier_init(self.value_proj, distribution='uniform', bias=0.)
        xavier_init(self.output_proj, distribution='uniform', bias=0.)    
        
    def get_sampling_offsets_and_attention(self, query):
        offsets = []
        attns = []
        #for i, (query, fc, attn) in enumerate(zip(queries, self.sampling_offsets, self.attention_weights)):
        #for i, query in enumerate(queries):
        bs, l, d = query.shape
        offset = self.sampling_offsets.to(query.device)(query).reshape(bs, l, self.num_heads, self.num_levels, self.num_points, -1, 2)
        offsets.append(offset)

        attention = self.attention_weights(query).reshape(bs, l, self.num_heads, self.num_levels, -1)

        level_attention = attention[:, :, :, :, -1:].softmax(-2) # bs, l, H, 3, 1
        attention = attention[:, :, :, :, :-1]
        attention = attention.softmax(-1) # bs, l, H, 3, p
        attention = attention * level_attention
        attns.append(attention)
  
        offsets = torch.cat(offsets, dim=1)
        attns = torch.cat(attns, dim=1)
    
        return offsets, attns

    def reshape_output(self, output, lens):
        bs, _, d = output.shape
        split_sizes = [lens[i] for i in range(self.num_slices)]
        outputs = torch.split(output, split_sizes, dim=1)        

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
        query_lens = [q.shape[1] for q in query]
        #split_sizes = 10
        ref_2d_s = self.get_reference_points(5, 5, dim='2d').to(query[0].device)
        #print(ref_2d_s.shape)
        spatial_shape = torch.tensor([[5, 5]], device=query[0].device)
        level_index = torch.tensor([0, 5*5], device=query[0].device)
        outputs = []
        #print(len(query))
        for q in query: 
            #outputs = torch.split(output, split_sizes, dim=1)
            bs, l, d = q.shape
            X, Y = 20, 20 
            a, b = 5, 5 
            final_shape = (2*20, 2*20)
            l_new = final_shape[0] * final_shape[1]
            #slide_voxel_feature = self.slide_embedding.unsqueeze(0).repeat(bs, 1, 1) #torch.zeros((bs, l_new, d))
            new_q_list = []
            for i in range(0, X - a + 1, a):
                for j in range(0, Y - b + 1, b):
                    # Compute the original indices in the flattened array
                    window_indices = [
                        (i + x) * Y + (j + y)
                        for x in range(a) for y in range(b)
                    ]
                    #print(q.shape, window_indices)
                    assert q.size(1) > max(window_indices), "window_indices exceeds q dimensions"
                    window = q[:, window_indices, :]
                    window_query = window.unsqueeze(2).repeat(1, 1, 3, 1) + self.height_embedding[None,None,:,:].to(window.dtype)
                    window_query = window_query.flatten(1, 2)
                    #print(window_query.shape)

                    query_ = self.forward_window(                
                                            window_query,
                                            window,
                                            identity=window_query,
                                            query_pos=None,
                                            reference_points=ref_2d_s,
                                            spatial_shapes=spatial_shape,
                                            level_start_index=level_index,
                                            **kwargs)
                    
                    new_q_list.append(query_.reshape(bs, 5*5, 3, d).flatten(1,2))
                    #print(window_query.shape)
            new_q = torch.cat(new_q_list, dim=1)
            outputs.append(new_q)

        return outputs

    def forward_window(self,                
                query,
                value,
                identity=None,
                query_pos=None,
                reference_points=None,
                spatial_shapes=None,
                level_start_index=None,
                **kwargs):
   
        #above to split window features
        
        value = self.value_proj(value) #[layer(v) for layer, v in zip(self.value_proj, value)]
 
        bs, num_value, _ = value.shape
        value = value.view(bs, num_value, self.num_heads, -1)

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

            sampling_offsets = sampling_offsets.squeeze(-2)
            
            if reference_points.shape[-1] == 2:
                offset_normalizer = torch.stack(
                    [spatial_shapes[..., 1], spatial_shapes[..., 0]], -1).to(query[0].device)

                sampling_locations = reference_points[:, :, None, :, None, :] \
                    + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
                #sampling_locations = reference_points[:, :, None, :, None, :] / offset_normalizer[None, None, None, :, None, :]
                #print("slide", torch.max(sampling_locations))
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
      
        #print("line261", output.shape, identity.shape)
        #output = self.dropout(output) #self.dropout(self.output_proj(output))
        #outputs = self.reshape_output(output, query_lens)
        #results = []
        #for out, layer, drop, residual in zip(outputs, self.output_proj, self.dropout, identity):
        #results.append(residual + drop(layer(out)))

        return output 