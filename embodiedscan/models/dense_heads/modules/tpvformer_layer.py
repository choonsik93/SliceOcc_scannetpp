import copy
import warnings

import torch
from mmengine import Config, DictAction
from mmcv.cnn import build_norm_layer
from mmengine.model import BaseModule, ModuleList
from mmengine.config import ConfigDict
from mmcv.cnn.bricks.transformer import build_feedforward_network, build_attention
from embodiedscan.registry import MODELS



@MODELS.register_module()
class TPVFormerLayer(BaseModule):
    """Base `TransformerLayer` for vision transformer.
    It can be built from `mmcv.ConfigDict` and support more flexible
    customization, for example, using any number of `FFN or LN ` and
    use different kinds of `attention` by specifying a list of `ConfigDict`
    named `attn_cfgs`. It is worth mentioning that it supports `prenorm`
    when you specifying `norm` as the first element of `operation_order`.
    More details about the `prenorm`: `On Layer Normalization in the
    Transformer Architecture <https://arxiv.org/abs/2002.04745>`_ .
    Args:
        attn_cfgs (list[`mmcv.ConfigDict`] | obj:`mmcv.ConfigDict` | None )):
            Configs for `self_attention` or `cross_attention` modules,
            The order of the configs in the list should be consistent with
            corresponding attentions in operation_order.
            If it is a dict, all of the attention modules in operation_order
            will be built with this config. Default: None.
        ffn_cfgs (list[`mmcv.ConfigDict`] | obj:`mmcv.ConfigDict` | None )):
            Configs for FFN, The order of the configs in the list should be
            consistent with corresponding ffn in operation_order.
            If it is a dict, all of the attention modules in operation_order
            will be built with this config.
        operation_order (tuple[str]): The execution order of operation
            in transformer. Such as ('self_attn', 'norm', 'ffn', 'norm').
            Support `prenorm` when you specifying first element as `norm`.
            Default：None.
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='LN').
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
        batch_first (bool): Key, Query and Value are shape
            of (batch, n, embed_dim)
            or (n, batch, embed_dim). Default to False.
    """

    def __init__(self,
                 attn_cfgs=None,
                 ffn_cfgs=dict(
                     type='FFN',
                     feedforward_channels=1024,
                     num_fcs=2,
                     ffn_drop=0.,
                     act_cfg=dict(type='ReLU', inplace=True),
                 ),
                 operation_order=None,
                 norm_cfg=dict(type='LN'),
                 init_cfg=None,
                 batch_first=True,
                 **kwargs):
        # import pdb; pdb.set_trace()
        deprecated_args = dict(
            feedforward_channels='feedforward_channels',
            ffn_dropout='ffn_drop',
            ffn_num_fcs='num_fcs')
        for ori_name, new_name in deprecated_args.items():
            if ori_name in kwargs:
                warnings.warn(
                    f'The arguments `{ori_name}` in BaseTransformerLayer '
                    f'has been deprecated, now you should set `{new_name}` '
                    f'and other FFN related arguments '
                    f'to a dict named `ffn_cfgs`. ')
                ffn_cfgs[new_name] = kwargs[ori_name]

        super().__init__(init_cfg)

        self.batch_first = batch_first

        num_attn = operation_order.count('self_attn') + operation_order.count(
            'cross_attn') #+ operation_order.count('window_attn')

        if isinstance(attn_cfgs, dict):
            attn_cfgs = [copy.deepcopy(attn_cfgs) for _ in range(num_attn)]
        else:
            assert num_attn == len(attn_cfgs), f'The length ' \
                f'of attn_cfg {num_attn} is ' \
                f'not consistent with the number of attention' \
                f'in operation_order {operation_order}.'
        #print(attn_cfgs, num_attn, operation_order)
        self.last_layer = False
        self.num_attn = num_attn
        self.operation_order = operation_order
        self.norm_cfg = norm_cfg
        self.pre_norm = operation_order[0] == 'norm'
        self.attentions = ModuleList()

        index = 0
        for operation_name in operation_order:
            if operation_name in ['self_attn', 'cross_attn', 'window_attn']:
                if 'batch_first' in attn_cfgs[index]:
                    assert self.batch_first == attn_cfgs[index]['batch_first']
                else:
                    attn_cfgs[index]['batch_first'] = self.batch_first
                attention = build_attention(attn_cfgs[index])
                # Some custom attentions used as `self_attn`
                # or `cross_attn` can have different behavior.
                attention.operation_name = operation_name
                self.attentions.append(attention)
                index += 1

        self.embed_dims = self.attentions[0].embed_dims

        self.ffns = ModuleList()
        num_ffns = operation_order.count('ffn')
        if isinstance(ffn_cfgs, dict):
            ffn_cfgs = ConfigDict(ffn_cfgs)
        if isinstance(ffn_cfgs, dict):
            ffn_cfgs = [copy.deepcopy(ffn_cfgs) for _ in range(num_ffns)]
        assert len(ffn_cfgs) == num_ffns
        for ffn_index in range(num_ffns):
            if 'embed_dims' not in ffn_cfgs[ffn_index]:
                ffn_cfgs[ffn_index]['embed_dims'] = self.embed_dims
            else:
                assert ffn_cfgs[ffn_index]['embed_dims'] == self.embed_dims

            self.ffns.append(build_feedforward_network(ffn_cfgs[ffn_index]))

        self.norms = ModuleList()
        num_norms = operation_order.count('norm')
        for _ in range(num_norms):
            self.norms.append(build_norm_layer(norm_cfg, self.embed_dims)[1])

    def forward(self,
                query,
                key=None,
                value=None,
                tpv_pos=None,
                ref_2d=None,
                tpv_h=None,
                tpv_w=None,
                tpv_z=None,
                spatial_shapes=None,
                level_start_index=None,
                reference_points_cams=None,
                tpv_masks=None,
                **kwargs):
        """
        **kwargs contains some specific arguments of attentions.

        Args:
            query (Tensor): The input query with shape
                [num_queries, bs, embed_dims] if
                self.batch_first is False, else
                [bs, num_queries embed_dims].
            key (Tensor): The key tensor with shape [num_keys, bs,
                embed_dims] if self.batch_first is False, else
                [bs, num_keys, embed_dims] .
            value (Tensor): The value tensor with same shape as `key`.
        Returns:
            Tensor: forwarded results with shape [num_queries, bs, embed_dims].
        """

        norm_index = 0
        attn_index = 0
        ffn_index = 0
        #print(len(query)) #16
        if self.operation_order[0] == 'cross_attn':
            query = torch.cat(query, dim=1)
        identity = query

        for layer in self.operation_order:
            # cross view hybrid attention
            if layer == 'self_attn':
                #print("self att")
                '''
                ss = torch.tensor([
                    [tpv_h, tpv_w],
                    [tpv_z, tpv_h],
                    [tpv_w, tpv_z]
                ], device=query[0].device)
                '''
                #hw_shape = torch.tensor([tpv_h, tpv_w]).repeat(4,1)
                #zh_shape = torch.tensor([tpv_z, tpv_h]).repeat(10,1)
                #wz_shape = torch.tensor([tpv_w, tpv_z]).repeat(10,1)
                #ss = torch.cat((hw_shape, zh_shape, wz_shape), dim=0)
                queries_floor = [query[i] for i in range(0, len(query), 2)]
                queries_ceiling = [query[i] for i in range(1, len(query), 2)]
                ss = torch.tensor([
                    [tpv_h, tpv_w]
                ], device=query[0].device)
                lsi = torch.tensor([
                    0, tpv_h*tpv_w, tpv_h*tpv_w+tpv_z*tpv_h
                ], device=query[0].device)

                if not isinstance(query, (list, tuple)):
                    raise ValueError("Cross-plane queries must be list.")
   
                query_floor = self.attentions[attn_index](
                    queries_floor, #query,
                    queries_ceiling,
                    identity if self.pre_norm else None,
                    query_pos=tpv_pos,
                    reference_points=ref_2d,
                    spatial_shapes=ss,
                    level_start_index=lsi,
                    **kwargs)

                query_ceiling = self.attentions[attn_index](
                    queries_ceiling, #query,
                    query_floor,
                    identity if self.pre_norm else None,
                    query_pos=tpv_pos,
                    reference_points=ref_2d,
                    spatial_shapes=ss,
                    level_start_index=lsi,
                    **kwargs)                                        
                attn_index += 1
            
                new_query = []
                f_index = 0
                c_index = 0           
                while f_index < len(query_floor) or c_index < len(query_ceiling):
                    if f_index < len(query_floor):
                        new_query.append(query_floor[f_index])
                        f_index += 1
                    if c_index < len(query_ceiling):
                        new_query.append(query_ceiling[c_index])
                        c_index += 1
                                
                query = torch.cat(new_query, dim=1)
                identity = query

            elif layer == 'norm':
                query = self.norms[norm_index](query)
                norm_index += 1

            # image cross attention
            elif layer == 'cross_attn':
                #print("cross att")
                # print("ICA query shape: ", query.shape) torch.Size([1, 2880, 256])
                # print("ICA key & value shape: ", key.shape, value.shape) torch.Size([10, 14400, 1, 256]) torch.Size([10, 14400, 1, 256])
                #print("TPV Masks shape:", tpv_masks[0].shape, tpv_masks[1].shape, tpv_masks[2].shape) 
                #TPV Masks shape: torch.Size([10, 1, 1600, 16]) torch.Size([10, 1, 640, 64]) torch.Size([10, 1, 640, 64])
                self.attentions[attn_index].thisdevice = query.device
                query = self.attentions[attn_index](
                    query,
                    key,
                    value,
                    identity if self.pre_norm else None,
                    spatial_shapes=spatial_shapes,
                    level_start_index=level_start_index,
                    reference_points_cams=reference_points_cams,
                    tpv_masks=tpv_masks,
                    **kwargs)
                attn_index += 1
                identity = query
            
            elif layer == 'window_attn':
                #print("window_attention")
                plane_sizes = [tpv_h*tpv_w] * self.attentions[-2].num_slices[0] * 2 + [tpv_h*tpv_z] * self.attentions[-2].num_slices[1] * 2 + [tpv_w*tpv_z] * self.attentions[-2].num_slices[2] * 2
                query = torch.split(query, plane_sizes, dim=1)
                self.attentions[attn_index].thisdevice = query[0].device
                query = self.attentions[attn_index](
                    query,
                    None,
                    None,
                    None,
                    spatial_shapes=None,
                    level_start_index=None,
                    reference_points_cams=None,
                    tpv_masks=None,
                    **kwargs)
                attn_index += 1
                identity = query
                query = torch.cat(query, dim=1)
                #print(query.shape)
                
            elif layer == 'ffn':
                ffn = self.ffns[ffn_index]
                query = ffn(query, identity if self.pre_norm else None)
                ffn_index += 1

        #query = torch.split(query, [tpv_h*tpv_w, tpv_z*tpv_h, tpv_w*tpv_z], dim=1)
        #for q in query:
        #    print(q.shape)
        #plane_sizes = [tpv_h*tpv_w] * self.attentions[-2].num_slices[0] * 2 + [tpv_h*tpv_z] * self.attentions[-2].num_slices[1] * 2 + [tpv_w*tpv_z] * self.attentions[-2].num_slices[2] * 2
        if not self.last_layer:
            plane_sizes = [tpv_h *tpv_w] * self.attentions[-1].num_slices[0] * 2
        else:
            plane_sizes = [tpv_h *tpv_w *3] * self.attentions[-2].num_slices[0] * 2
        query = torch.split(query, plane_sizes, dim=1)


        #for q in query:
            #print(q.shape)        
        return query