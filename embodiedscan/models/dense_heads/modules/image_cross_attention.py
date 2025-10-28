
from mmcv.ops.multi_scale_deform_attn import multi_scale_deformable_attn_pytorch
import warnings
import torch
import torch.nn as nn
from mmengine.model import xavier_init, constant_init
#from mmcv.cnn.bricks.registry import ATTENTION
from mmcv.cnn.bricks.transformer import build_attention
import math
#from mmcv.runner import force_fp32, auto_fp16
from mmengine.model import BaseModule
from mmcv.utils import ext_loader
from .multi_scale_deformable_attn_function import MultiScaleDeformableAttnFunction_fp32
from embodiedscan.registry import MODELS
from torch_geometric.nn import GATConv

ext_module = ext_loader.load_ext(
    '_ext', ['ms_deform_attn_backward', 'ms_deform_attn_forward'])
import random
def probability_function(p):

    return random.random() < p




@MODELS.register_module()
class TPVImageCrossAttention(BaseModule):
    """An attention module used in TPVFormer.
    Args:
        embed_dims (int): The embedding dimension of Attention.
            Default: 256.
        num_cams (int): The number of cameras
        dropout (float): A Dropout layer on `inp_residual`.
            Default: 0..
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
        deformable_attention: (dict): The config for the deformable attention used in SCA.
    """

    def __init__(self,
                 embed_dims=256,
                 num_cams=6,
                 num_slices=4,
                 pc_range=None,
                 dropout=0.1,
                 init_cfg=None,
                 batch_first=False,
                 thisdevice=None,
                 deformable_attention=dict(
                     type='MSDeformableAttention3D',
                     embed_dims=256,
                     num_levels=4,
                     device=None),
                 tpv_h=None,
                 tpv_w=None,
                 tpv_z=None,
                 **kwargs
                 ):
        super().__init__(init_cfg)

        self.init_cfg = init_cfg
        self.dropout = nn.Dropout(dropout)
        self.pc_range = pc_range
        self.fp16_enabled = False
        self.deformable_attention = build_attention(deformable_attention)
        self.deformable_attention.device = thisdevice
        #print(self.get_device(self.deformable_attention))
        self.embed_dims = embed_dims
        self.num_cams = num_cams
        self.output_proj = nn.Linear(embed_dims, embed_dims)
        self.batch_first = batch_first
        self.tpv_h, self.tpv_w, self.tpv_z = tpv_h, tpv_w, tpv_z
        self.init_weight()
        self.num_slices = num_slices
        self.graph_attention_networks = nn.ModuleList([GATConv(in_channels=embed_dims, out_channels=embed_dims, heads=1) for _ in range(2 * self.num_slices[0])])

    def init_weight(self):
        """Default initialization for Parameters of Module."""
        xavier_init(self.output_proj, distribution='uniform', bias=0.)

    def get_device(self, module):
        return next(module.parameters()).device

    def forward(self,
                query,
                key,
                value,
                residual=None,
                spatial_shapes=None,
                reference_points_cams=None,
                tpv_masks=None,
                level_start_index=None,
                **kwargs):
        """Forward Function of Detr3DCrossAtten.
        Args:
            query (Tensor): Query of Transformer with shape
                (bs, num_query, embed_dims).
            key (Tensor): The key tensor with shape
                (bs, num_key, embed_dims).
            value (Tensor): The value tensor with shape
                (bs, num_key, embed_dims).
            residual (Tensor): The tensor used for addition, with the
                same shape as `x`. Default None. If None, `x` will be used.
            spatial_shapes (Tensor): Spatial shape of features in
                different level. With shape  (num_levels, 2),
                last dimension represent (h, w).
            level_start_index (Tensor): The start index of each level.
                A tensor has shape (num_levels) and can be represented
                as [0, h_0*w_0, h_0*w_0+h_1*w_1, ...].
        Returns:
             Tensor: forwarded results with shape [num_query, bs, embed_dims].
        """
        if key is None:
            key = query
        if value is None:
            value = key
        
        if residual is None:
            inp_residual = query
        
        bs, num_query, _ = query.size()
        plane_sizes = [self.tpv_h*self.tpv_w] * self.num_slices[0] * 2 + [self.tpv_h*self.tpv_z] * self.num_slices[1] * 2 + [self.tpv_w*self.tpv_z] * self.num_slices[2] * 2
        queries = torch.split(query, plane_sizes, dim=1)
        
        if residual is None:
            slots = [torch.zeros_like(q) for q in queries]
        indexeses = []
        max_lens = []
        queries_rebatches = []
        reference_points_rebatches = []

        max_length = 900 #950  
        for tpv_idx, tpv_mask in enumerate(tpv_masks):
            indexes = []
            for _, mask_per_img in enumerate(tpv_mask):
                index_query_per_img = mask_per_img[0].sum(-1).nonzero().squeeze(-1)
                
                if index_query_per_img.size(0) > max_length:
                    indices = torch.arange(index_query_per_img.size(0))
                    sampled_indices = torch.randperm(index_query_per_img.size(0))[:max_length]
                    sampled_indices = torch.sort(sampled_indices)[0]
                    index_query_per_img = index_query_per_img[sampled_indices]
                
                indexes.append(index_query_per_img)
            max_len = max([len(each) for each in indexes])
            max_lens.append(max_len)
            indexeses.append(indexes)
            reference_points_cam = reference_points_cams[tpv_idx]
            D = reference_points_cam.size(3)

            queries_rebatch = queries[tpv_idx].new_zeros(
                [bs * self.num_cams, max_len, self.embed_dims])
            reference_points_rebatch = reference_points_cam.new_zeros(
                [bs * self.num_cams, max_len, D, 2])


            for i, reference_points_per_img in enumerate(reference_points_cam):
                for j in range(bs):
                    index_query_per_img = indexes[i]
                    #print(j, j * self.num_cams + i, tpv_idx)
                    queries_rebatch[j * self.num_cams + i, :len(index_query_per_img)] = queries[tpv_idx][j, index_query_per_img]
                    reference_points_rebatch[j * self.num_cams + i, :len(index_query_per_img)] = reference_points_per_img[j, index_query_per_img]
         
            queries_rebatches.append(queries_rebatch)
            reference_points_rebatches.append(reference_points_rebatch)

        num_cams, l, bs, embed_dims = key.shape

        key = key.permute(0, 2, 1, 3).view(
            self.num_cams * bs, l, self.embed_dims)
        value = value.permute(0, 2, 1, 3).view(
            self.num_cams * bs, l, self.embed_dims)

        queries = self.deformable_attention(
            query=queries_rebatches, key=key, value=value,
            reference_points=reference_points_rebatches, 
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,)
        
        for tpv_idx, indexes in enumerate(indexeses):
            for i, index_query_per_img in enumerate(indexes):
                for j in range(bs):
                    slots[tpv_idx][j, index_query_per_img] += queries[tpv_idx][j * self.num_cams + i, :len(index_query_per_img)]

            count = tpv_masks[tpv_idx].sum(-1) > 0
            count = count.permute(1, 2, 0).sum(-1)
            count = torch.clamp(count, min=1.0)
            slots[tpv_idx] = slots[tpv_idx] / count[..., None]
        slots = torch.cat(slots, dim=1)
        slots = self.output_proj(slots)

        return self.dropout(slots) + inp_residual


    def get_nearest_neighbors(self, center_indices, all_indices, query_spatial_shape, k=4):
        """
        Get the indices of the nearest neighbors for each center index.
        """
        h, w = query_spatial_shape
        center_coords = torch.stack([center_indices // w, center_indices % w], dim=-1)
        all_coords = torch.stack([all_indices // w, all_indices % w], dim=-1)

        # Calculate distances between center coordinates and all other coordinates
        distances = ((center_coords.unsqueeze(1) - all_coords.unsqueeze(0)) ** 2).sum(dim=-1)
        
        # Find the indices of the k-nearest neighbors
        _, sorted_indices = torch.topk(distances, k + 1, dim=-1, largest=False)
        
        # Exclude the center point itself from the neighbors
        nearest_neighbors = sorted_indices[:, 1:]
        
        # Convert back to linear indices
        nearest_neighbors_linear = nearest_neighbors[:, :, 0] * w + nearest_neighbors[:, :, 1]
        
        return nearest_neighbors_linear


    def build_edge_index_batch(self, center_indices, nearest_neighbors, num_nodes):
        """
        Builds the edge_index tensor for a batch of graphs.
        """
        # Flatten the tensors to make it easier to work with them
        center_indices_flat = center_indices.squeeze(1)  # Remove the extra dimension
        nearest_neighbors_flat = nearest_neighbors.reshape(-1)  # Flatten the nearest neighbors
        
        # Create the edge_index tensor
        edge_index = torch.stack([center_indices_flat.repeat_interleave(nearest_neighbors.size(1)),
                                nearest_neighbors_flat], dim=0)
        
        # Reshape the edge_index to include both directions (i -> j and j -> i)
        edge_index = torch.cat([edge_index, edge_index[[1, 0]]], dim=1)

        # Since num_nodes is fixed for each graph, we do not need to shift the indices
        # for different graphs in this case.

        return edge_index


@MODELS.register_module()
class TPVMSDeformableAttention3D(BaseModule):
    """An attention module used in tpvFormer based on Deformable-Detr.
    `Deformable DETR: Deformable Transformers for End-to-End Object Detection.
    <https://arxiv.org/pdf/2010.04159.pdf>`_.
    Args:
        embed_dims (int): The embedding dimension of Attention.
            Default: 256.
        num_heads (int): Parallel attention heads. Default: 64.
        num_levels (int): The number of feature map used in
            Attention. Default: 4.
        num_points (int): The number of sampling points for
            each query in each head. Default: 4.
        im2col_step (int): The step used in image_to_column.
            Default: 64.
        dropout (float): A Dropout layer on `inp_identity`.
            Default: 0.1.
        batch_first (bool): Key, Query and Value are shape of
            (batch, n, embed_dim)
            or (n, batch, embed_dim). Default to False.
        norm_cfg (dict): Config dict for normalization layer.
            Default: None.
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
    """

    def __init__(self,
                 embed_dims=256,
                 num_heads=8,
                 num_levels=4,
                 num_points=[16, 16, 16, 16],
                 num_slices=4,
                 num_z_anchors=[8, 8, 8, 8],
                 pc_range=None,
                 im2col_step=64,
                 dropout=0.1,
                 batch_first=True,
                 norm_cfg=None,
                 init_cfg=None,
                 floor_sampling_offset=True,
                 device=None,
                 tpv_h=None,
                 tpv_w=None,
                 tpv_z=None,
                ):
        super().__init__(init_cfg)
        if embed_dims % num_heads != 0:
            raise ValueError(f'embed_dims must be divisible by num_heads, '
                             f'but got {embed_dims} and {num_heads}')
        dim_per_head = embed_dims // num_heads
        self.norm_cfg = norm_cfg
        self.batch_first = batch_first
        self.output_proj = None
        self.fp16_enabled = False

        # you'd better set dim_per_head to a power of 2
        # which is more efficient in the CUDA implementation
        def _is_power_of_2(n):
            if (not isinstance(n, int)) or (n < 0):
                raise ValueError(
                    'invalid input for _is_power_of_2: {} (type: {})'.format(
                        n, type(n)))
            return (n & (n - 1) == 0) and n != 0

        if not _is_power_of_2(dim_per_head):
            warnings.warn(
                "You'd better set embed_dims in "
                'MultiScaleDeformAttention to make '
                'the dimension of each attention head a power of 2 '
                'which is more efficient in our CUDA implementation.')

        self.im2col_step = im2col_step
        self.embed_dims = embed_dims
        self.num_levels = num_levels
        self.num_heads = num_heads
        self.num_points = num_points
        self.num_z_anchors = num_z_anchors
        self.base_num_points = num_points[0]
        self.base_z_anchors = num_z_anchors[0]
        self.points_multiplier = [points // self.base_z_anchors for points in num_z_anchors]
        self.device = device
        self.num_slices = num_slices

        self.sampling_offsets = nn.ModuleList([
            nn.Linear(embed_dims, num_heads * num_levels * num_points[i] * 2) for i in range(2 * self.num_slices) 
        ])
        self.floor_sampling_offset = floor_sampling_offset
        self.attention_weights = nn.ModuleList([
            nn.Linear(embed_dims, num_heads * num_levels * num_points[i]) for i in range(2 * self.num_slices)
        ])
        self.value_proj = nn.Linear(embed_dims, embed_dims)

        self.init_weights()

    def init_weights(self):
        """Default initialization for Parameters of Module."""
        for i in range(2 * self.num_slices): # num planes
            constant_init(self.sampling_offsets[i], 0.)
            thetas = torch.arange(
                self.num_heads,
                dtype=torch.float32) * (2.0 * math.pi / self.num_heads)
            grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
            grid_init = (grid_init /
                        grid_init.abs().max(-1, keepdim=True)[0]).view(
                self.num_heads, 1, 1,
                2).repeat(1, self.num_levels, self.num_points[i], 1)
            #print(grid_init.shape)
            grid_init = grid_init.reshape(self.num_heads, self.num_levels, self.num_z_anchors[i], -1, 2)
            for j in range(self.num_points[i] // self.num_z_anchors[i]):
                grid_init[:, :, :, j, :] *= j + 1
 
            self.sampling_offsets[i].bias.data = grid_init.view(-1)

            constant_init(self.attention_weights[i], val=0., bias=0.)
        xavier_init(self.value_proj, distribution='uniform', bias=0.)
        xavier_init(self.output_proj, distribution='uniform', bias=0.)
        self._is_init = True

    def get_sampling_offsets_and_attention(self, queries):
        offsets = []
        attns = []
        for i, (query, fc, attn) in enumerate(zip(queries, self.sampling_offsets, self.attention_weights)):
            bs, l, d = query.shape
            #print(bs, l, d)
            offset = fc.to(query.device)(query).reshape(bs, l, self.num_heads, self.num_levels, self.points_multiplier[i], -1, 2)
            offset = offset.permute(0, 1, 4, 2, 3, 5, 6).flatten(1, 2)
            offsets.append(offset)

            attention = attn(query).reshape(bs, l, self.num_heads, -1)
            attention = attention.softmax(-1)
            attention = attention.view(bs, l, self.num_heads, self.num_levels, self.points_multiplier[i], -1)
            attention = attention.permute(0, 1, 4, 2, 3, 5).flatten(1, 2)
            attns.append(attention)
        
        offsets = torch.cat(offsets, dim=1)
        attns = torch.cat(attns, dim=1)
        return offsets, attns

    def reshape_reference_points(self, reference_points):
        reference_point_list = []
        for i, reference_point in enumerate(reference_points):
            bs, l, z_anchors, _  = reference_point.shape
            reference_point = reference_point.reshape(bs, l, self.points_multiplier[i], -1, 2)
            reference_point = reference_point.flatten(1, 2)
            reference_point_list.append(reference_point)
        return torch.cat(reference_point_list, dim=1)
    
    def reshape_output(self, output, lens):
        bs, _, d = output.shape
        #outputs = torch.split(output, [lens[0]*self.points_multiplier[0], lens[1]*self.points_multiplier[1], 
        #lens[2]*self.points_multiplier[2], lens[3]*self.points_multiplier[3]], dim=1)
        #print(len(self.points_multiplier), len(lens))
        split_sizes = [lens[i] * self.points_multiplier[i] for i in range(2 * self.num_slices)]
        outputs = torch.split(output, split_sizes, dim=1)

        outputs = [o.reshape(bs, -1, self.points_multiplier[i], d).sum(dim=2) for i, o in enumerate(outputs)]
        return outputs

    def forward(self,
                query,
                key=None,
                value=None,
                identity=None,
                reference_points=None,
                spatial_shapes=None,
                level_start_index=None,
                **kwargs):
        """Forward Function of MultiScaleDeformAttention.
        Args:
            query (Tensor): Query of Transformer with shape
                ( bs, num_query, embed_dims).
            key (Tensor): The key tensor with shape
                `(bs, num_key,  embed_dims)`.
            value (Tensor): The value tensor with shape
                `(bs, num_key,  embed_dims)`.
            identity (Tensor): The tensor used for addition, with the
                same shape as `query`. Default None. If None,
                `query` will be used.
            reference_points (Tensor):  The normalized reference
                points with shape (bs, num_query, num_levels, 2),
                all elements is range in [0, 1], top-left (0,0),
                bottom-right (1, 1), including padding area.
                or (N, Length_{query}, num_levels, 4), add
                additional two dimensions is (w, h) to
                form reference boxes.
            spatial_shapes (Tensor): Spatial shape of features in
                different levels. With shape (num_levels, 2),
                last dimension represents (h, w).
            level_start_index (Tensor): The start index of each level.
                A tensor has shape ``(num_levels, )`` and can be represented
                as [0, h_0*w_0, h_0*w_0+h_1*w_1, ...].
        Returns:
             Tensor: forwarded results with shape [bs, num_query, embed_dims].
        """

        if value is None:
            value = query
        if identity is None:
            identity = query
        
        if not self.batch_first:
            # change to (bs, num_query ,embed_dims)
            query = [q.permute(1, 0, 2) for q in query]
            value = value.permute(1, 0, 2)

        # bs, num_query, _ = query.shape
        #print(len(query)) #8
        #for q in query:
            #print(q.shape) 
        query_lens = [q.shape[1] for q in query]
        bs, num_value, _ = value.shape
        assert (spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum() == num_value

        value = self.value_proj(value)
        value = value.view(bs, num_value, self.num_heads, -1)
        sampling_offsets, attention_weights = self.get_sampling_offsets_and_attention(query)
        #print(reference_points[0].shape, reference_points[1].shape, reference_points[2].shape, reference_points[3].shape, reference_points[4].shape, reference_points[5].shape, len(reference_points))
        reference_points = self.reshape_reference_points(reference_points)
        
        if reference_points.shape[-1] == 2:
            """
            For each tpv query, it owns `num_Z_anchors` in 3D space that having different heights.
            After proejcting, each tpv query has `num_Z_anchors` reference points in each 2D image.
            For each referent point, we sample `num_points` sampling points.
            For `num_Z_anchors` reference points,  it has overall `num_points * num_Z_anchors` sampling points.
            """
            offset_normalizer = torch.stack(
                [spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)

            bs, num_query, num_Z_anchors, xy = reference_points.shape
            reference_points = reference_points[:, :, None, None, :, None, :].to(sampling_offsets.device)
            #print(sampling_offsets.shape, offset_normalizer.shape) #torch.Size([20, 4075, 8, 4, 6, 2]) torch.Size([1, 2])
            sampling_offsets = sampling_offsets / \
                offset_normalizer[None, None, None, :, None, :]
            bs, num_query, num_heads, num_levels, num_all_points, xy = sampling_offsets.shape
            sampling_offsets = sampling_offsets.view(
                bs, num_query, num_heads, num_levels, num_Z_anchors, num_all_points // num_Z_anchors, xy)
            sampling_locations = reference_points + sampling_offsets
            bs, num_query, num_heads, num_levels, num_points, num_Z_anchors, xy = sampling_locations.shape
            assert num_all_points == num_points * num_Z_anchors

            sampling_locations = sampling_locations.view(
                bs, num_query, num_heads, num_levels, num_all_points, xy)
            
            if self.floor_sampling_offset:
                sampling_locations = sampling_locations - torch.floor(sampling_locations)

        elif reference_points.shape[-1] == 4:
            assert False
        else:
            raise ValueError(
                f'Last dim of reference_points must be'
                f' 2 or 4, but get {reference_points.shape[-1]} instead.')

        #  sampling_locations.shape: bs, num_query, num_heads, num_levels, num_all_points, 2
        #  attention_weights.shape: bs, num_query, num_heads, num_levels, num_all_points

        if torch.cuda.is_available() and value.is_cuda:
            if value.dtype == torch.float16:
                MultiScaleDeformableAttnFunction = MultiScaleDeformableAttnFunction_fp32
            else:
                MultiScaleDeformableAttnFunction = MultiScaleDeformableAttnFunction_fp32
            output = MultiScaleDeformableAttnFunction.apply(
                value, spatial_shapes, level_start_index, sampling_locations,
                attention_weights, self.im2col_step)
        else:
            output = multi_scale_deformable_attn_pytorch(
                value, spatial_shapes, sampling_locations, attention_weights)

        output = self.reshape_output(output, query_lens)
        if not self.batch_first:
            output = [o.permute(1, 0, 2) for o in output]
        #for o in output:
        #    print(o.shape)
        return output