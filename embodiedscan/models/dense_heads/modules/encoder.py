
from mmcv.cnn.bricks.transformer import TransformerLayerSequence
#from mmcv.runner import force_fp32, auto_fp16
import numpy as np
import torch
from mmengine.utils.dl_utils import TORCH_VERSION
from mmcv.utils import ext_loader
ext_module = ext_loader.load_ext(
    '_ext', ['ms_deform_attn_backward', 'ms_deform_attn_forward'])
from embodiedscan.registry import MODELS
from embodiedscan.structures.bbox_3d import (batch_points_cam2img,
                                             get_proj_mat_by_coord_type,
                                             points_cam2img, points_img2cam)

from embodiedscan.models.layers.fusion_layers.point_fusion import apply_3d_transformation

@MODELS.register_module()
class SliceOccEncoder(TransformerLayerSequence):

    """
    Attention with both self and cross attention.
    Args:
        return_intermediate (bool): Whether to return intermediate outputs.
        coder_norm_cfg (dict): Config of last normalization layer. Default: `LN`.
    """

    def __init__(self, *args, slice_h, slice_w, slice_z, pc_range=None, 
                 num_points_in_pillar=[4, 32, 32], slice_num=4,
                 num_points_in_pillar_cross_view=[32, 32, 32],
                 return_intermediate=False, **kwargs):

        super().__init__(*args, **kwargs)
        self.return_intermediate = return_intermediate
        self.slice_num = slice_num
        self.slice_h, self.slice_w, self.slice_z = slice_h, slice_w, slice_z
        self.num_points_in_pillar = num_points_in_pillar
        #assert num_points_in_pillar[1] == num_points_in_pillar[2] and num_points_in_pillar[1] % num_points_in_pillar[0] == 0
        self.pc_range = pc_range
        self.fp16_enabled = False

        
        #cross_view_ref_points = self.get_cross_view_ref_points(slice_h, slice_w, slice_z, num_points_in_pillar_cross_view)
        #self.register_buffer('cross_view_ref_points', cross_view_ref_points)
        self.num_points_cross_view = num_points_in_pillar_cross_view


    @staticmethod
    def get_cross_view_ref_points(slice_h, slice_w, slice_z, num_points_in_pillar):
        # ref points generating target: (#query)hw+zh+wz, (#level)3, #p, 2
        # generate points for hw and level 1
        h_ranges = torch.linspace(0.5, slice_h-0.5, slice_h) / slice_h
        w_ranges = torch.linspace(0.5, slice_w-0.5, slice_w) / slice_w
        h_ranges = h_ranges.unsqueeze(-1).expand(-1, slice_w).flatten()
        w_ranges = w_ranges.unsqueeze(0).expand(slice_h, -1).flatten()
        hw_hw = torch.stack([w_ranges, h_ranges], dim=-1) # hw, 2
        hw_hw = hw_hw.unsqueeze(1).expand(-1, num_points_in_pillar[2], -1) # hw, #p, 2
        # generate points for hw and level 2
        z_ranges = torch.linspace(0.5, slice_z-0.5, num_points_in_pillar[2]) / slice_z # #p
        z_ranges = z_ranges.unsqueeze(0).expand(slice_h*slice_w, -1) # hw, #p
        h_ranges = torch.linspace(0.5, slice_h-0.5, slice_h) / slice_h
        h_ranges = h_ranges.reshape(-1, 1, 1).expand(-1, slice_w, num_points_in_pillar[2]).flatten(0, 1)
        hw_zh = torch.stack([h_ranges, z_ranges], dim=-1) # hw, #p, 2
        # generate points for hw and level 3
        z_ranges = torch.linspace(0.5, slice_z-0.5, num_points_in_pillar[2]) / slice_z # #p
        z_ranges = z_ranges.unsqueeze(0).expand(slice_h*slice_w, -1) # hw, #p
        w_ranges = torch.linspace(0.5, slice_w-0.5, slice_w) / slice_w
        w_ranges = w_ranges.reshape(1, -1, 1).expand(slice_h, -1, num_points_in_pillar[2]).flatten(0, 1)
        hw_wz = torch.stack([z_ranges, w_ranges], dim=-1) # hw, #p, 2
        
        # generate points for zh and level 1
        w_ranges = torch.linspace(0.5, slice_w-0.5, num_points_in_pillar[1]) / slice_w
        w_ranges = w_ranges.unsqueeze(0).expand(slice_z*slice_h, -1)
        h_ranges = torch.linspace(0.5, slice_h-0.5, slice_h) / slice_h
        h_ranges = h_ranges.reshape(1, -1, 1).expand(slice_z, -1, num_points_in_pillar[1]).flatten(0, 1)
        zh_hw = torch.stack([w_ranges, h_ranges], dim=-1)
        # generate points for zh and level 2
        z_ranges = torch.linspace(0.5, slice_z-0.5, slice_z) / slice_z
        z_ranges = z_ranges.reshape(-1, 1, 1).expand(-1, slice_h, num_points_in_pillar[1]).flatten(0, 1)
        h_ranges = torch.linspace(0.5, slice_h-0.5, slice_h) / slice_h
        h_ranges = h_ranges.reshape(1, -1, 1).expand(slice_z, -1, num_points_in_pillar[1]).flatten(0, 1)
        zh_zh = torch.stack([h_ranges, z_ranges], dim=-1) # zh, #p, 2
        # generate points for zh and level 3
        w_ranges = torch.linspace(0.5, slice_w-0.5, num_points_in_pillar[1]) / slice_w
        w_ranges = w_ranges.unsqueeze(0).expand(slice_z*slice_h, -1)
        z_ranges = torch.linspace(0.5, slice_z-0.5, slice_z) / slice_z
        z_ranges = z_ranges.reshape(-1, 1, 1).expand(-1, slice_h, num_points_in_pillar[1]).flatten(0, 1)
        zh_wz = torch.stack([z_ranges, w_ranges], dim=-1)

        # generate points for wz and level 1
        h_ranges = torch.linspace(0.5, slice_h-0.5, num_points_in_pillar[0]) / slice_h
        h_ranges = h_ranges.unsqueeze(0).expand(slice_w*slice_z, -1)
        w_ranges = torch.linspace(0.5, slice_w-0.5, slice_w) / slice_w
        w_ranges = w_ranges.reshape(-1, 1, 1).expand(-1, slice_z, num_points_in_pillar[0]).flatten(0, 1)
        wz_hw = torch.stack([w_ranges, h_ranges], dim=-1)
        # generate points for wz and level 2
        h_ranges = torch.linspace(0.5, slice_h-0.5, num_points_in_pillar[0]) / slice_h
        h_ranges = h_ranges.unsqueeze(0).expand(slice_w*slice_z, -1)
        z_ranges = torch.linspace(0.5, slice_z-0.5, slice_z) / slice_z
        z_ranges = z_ranges.reshape(1, -1, 1).expand(slice_w, -1, num_points_in_pillar[0]).flatten(0, 1)
        wz_zh = torch.stack([h_ranges, z_ranges], dim=-1)
        # generate points for wz and level 3
        w_ranges = torch.linspace(0.5, slice_w-0.5, slice_w) / slice_w
        w_ranges = w_ranges.reshape(-1, 1, 1).expand(-1, slice_z, num_points_in_pillar[0]).flatten(0, 1)
        z_ranges = torch.linspace(0.5, slice_z-0.5, slice_z) / slice_z
        z_ranges = z_ranges.reshape(1, -1, 1).expand(slice_w, -1, num_points_in_pillar[0]).flatten(0, 1)
        wz_wz = torch.stack([z_ranges, w_ranges], dim=-1)

        reference_points = torch.cat([
            torch.stack([hw_hw, hw_zh, hw_wz], dim=1),
            torch.stack([zh_hw, zh_zh, zh_wz], dim=1),
            torch.stack([wz_hw, wz_zh, wz_wz], dim=1)
        ], dim=0) 
        
        return reference_points

    @staticmethod
    def get_reference_points(H, W, Z=2.56, num_points_in_pillar=4, dim='3d', bs=1, device='cuda', dtype=torch.float, slice_order=None, axis=None):

        # reference points in 3D space, used in spatial cross-attention (SCA)
        if dim == '3d':
            segment_size = Z / len(slice_order)
            ref_3d_list = []
            for z_idx in slice_order:
                z_begin = z_idx * segment_size
                z_end = (z_idx + 1.0) * segment_size             
                slice_height = (z_end - z_begin) / 2.0
     
                zs_up = torch.linspace(z_begin + slice_height - 0.5, z_end, num_points_in_pillar, dtype=dtype, 
                                  device=device).view(-1, 1, 1).expand(num_points_in_pillar, H, W) / Z
                zs_low = torch.linspace(z_begin, z_end - slice_height + 0.5, num_points_in_pillar, dtype=dtype, 
                                  device=device).view(-1, 1, 1).expand(num_points_in_pillar, H, W) / Z    

                xs = torch.linspace(0.5, W - 0.5, W, dtype=dtype,
                                    device=device).view(1, 1, -1).expand(num_points_in_pillar, H, W) / W
                ys = torch.linspace(0.5, H - 0.5, H, dtype=dtype,
                                    device=device).view(1, -1, 1).expand(num_points_in_pillar, H, W) / H

                ref_3d_up = torch.stack((xs, ys, zs_up), -1)
                ref_3d_up = ref_3d_up.permute(0, 3, 1, 2).flatten(2).permute(0, 2, 1)
                ref_3d_up = ref_3d_up[None].repeat(bs, 1, 1, 1)

                ref_3d_low = torch.stack((xs, ys, zs_low), -1)
                ref_3d_low = ref_3d_low.permute(0, 3, 1, 2).flatten(2).permute(0, 2, 1)
                ref_3d_low = ref_3d_low[None].repeat(bs, 1, 1, 1)

                #print(torch.max(ref_3d_low[:,:,:,0]), torch.max(ref_3d_low[:,:,:,1]), torch.max(ref_3d_low[:,:,:,2]))
                #print(torch.min(ref_3d_low[:,:,:,0]), torch.min(ref_3d_low[:,:,:,1]), torch.min(ref_3d_low[:,:,:,2]))

                ref_3d_list.append(ref_3d_low)
                ref_3d_list.append(ref_3d_up)
            
            return ref_3d_list

        elif dim == '2d':
            ref_2d_list = []
            for i, z_idx in enumerate(slice_order):
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
                #print(ref_2d.shape)
            return ref_2d_list

    def point_sampling(self, reference_points, pc_range,  img_metas):

        lidar2img = []
        for img_meta in img_metas:
            #tmp = [img_meta['depth2img']['intrinsic'][i].detach().cpu() @ img_meta['depth2img']['extrinsic'][i].detach().cpu() for i in range(len(img_meta['depth2img']['extrinsic']))]
            tmp = [img_meta['depth2img']['intrinsic'][i] @ img_meta['depth2img']['extrinsic'][i] for i in range(len(img_meta['depth2img']['extrinsic']))]
            lidar2img.append(tmp)
    
        lidar2img = np.asarray(lidar2img)
        lidar2img = reference_points.new_tensor(lidar2img)  # (B, N, 4, 4)
        #lidar2img = torch.randn(1, 10, 4, 4).to(reference_points.device)
        reference_points = reference_points.clone()
        
        reference_points[..., 0:1] = reference_points[..., 0:1] * \
            (pc_range[3] - pc_range[0]) + pc_range[0]
        reference_points[..., 1:2] = reference_points[..., 1:2] * \
            (pc_range[4] - pc_range[1]) + pc_range[1]
        reference_points[..., 2:3] = reference_points[..., 2:3] * \
            (pc_range[5] - pc_range[2]) + pc_range[2]

        reference_points = torch.cat(
            (reference_points, torch.ones_like(reference_points[..., :1])), -1)

        reference_points = reference_points.permute(1, 0, 2, 3)
        D, B, num_query = reference_points.size()[:3]
        num_cam = lidar2img.size(1)

        reference_points = reference_points.view(
            D, B, 1, num_query, 4).repeat(1, 1, num_cam, 1, 1).unsqueeze(-1)

        lidar2img = lidar2img.view(
            1, B, num_cam, 1, 4, 4).repeat(D, 1, 1, num_query, 1, 1)

        reference_points_cam = torch.matmul(lidar2img.to(torch.float32),
                                            reference_points.to(torch.float32)).squeeze(-1)
        eps = 1e-5

        volume_mask = (reference_points_cam[..., 2:3] > eps)
        reference_points_cam = reference_points_cam[..., 0:2] / torch.maximum(
            reference_points_cam[..., 2:3], torch.ones_like(reference_points_cam[..., 2:3]) * eps)

        # _BUG FIXED!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        ori_H, ori_W = img_metas[0]['ori_shape'][:2]
        reference_points_cam[..., 0] /= ori_W
        reference_points_cam[..., 1] /= ori_H
        reference_points_cam[..., 0] /= 480
        reference_points_cam[..., 1] /= 480
        
        volume_mask = (volume_mask & (reference_points_cam[..., 1:2] > 0.0)
                    & (reference_points_cam[..., 1:2] < 1.0)
                    & (reference_points_cam[..., 0:1] < 1.0)
                    & (reference_points_cam[..., 0:1] > 0.0))
        
        volume_mask = torch.nan_to_num(volume_mask)
        
        reference_points_cam = reference_points_cam.permute(2, 1, 3, 0, 4) #num_cam, B, num_query, D, 3
        volume_mask = volume_mask.permute(2, 1, 3, 0, 4).squeeze(-1)

        return reference_points_cam, volume_mask
    

    def forward(self,
                slice_query, # list
                key,
                value,
                *args,
                slice_order=None,
                slice_h=None,
                slice_w=None,
                slice_z=None,
                slice_pos=None, # list
                spatial_shapes=None,
                level_start_index=None,
                **kwargs):
        """Forward function for `TransformerDecoder`.
        Args:
            slice_query (Tensor): Input slice query with shape
                `(num_query, bs, embed_dims)`.
            key & value (Tensor): Input multi-cameta features with shape
                (num_cam, num_value, bs, embed_dims)
            reference_points (Tensor): The reference
                points of offset. has shape
                (bs, num_query, 4) when as_two_stage,
                otherwise has shape ((bs, num_query, 2).
        """
        output = slice_query
        intermediate = []
        bs = slice_query[0].shape[0]

        reference_points_cams, slice_masks = [], []
        
        # ref_3ds_z = self.get_reference_points(slice_h, slice_w, 16, self.num_points_in_pillar[0], '3d', device='cpu', slice_order=slice_order[0], axis='z')
        # Change for scannetpp, scannet++
        ref_3ds_z = self.get_reference_points(slice_h, slice_w, slice_z // 2, self.num_points_in_pillar[0], '3d', device='cpu', slice_order=slice_order[0], axis='z')
        ref_3ds = ref_3ds_z #+ ref_3ds_h + ref_3ds_w

        ref_2ds_z = self.get_reference_points(slice_h, slice_w, self.pc_range[5]-self.pc_range[2], self.num_points_in_pillar[0], '2d', device='cpu', slice_order=slice_order[0], axis='z')
        ref_2ds = ref_2ds_z  
        ref_2ds = torch.cat(ref_2ds, dim=1).to(slice_query[0].device)
  
        
        for ref_idx, ref_3d in enumerate(ref_3ds):
            reference_points_cam, slice_mask = self.point_sampling(
                ref_3d, self.pc_range, kwargs['img_metas']) # num_cam, bs, hw++, #p, 2
            
            if not slice_mask.any():  
                random_indices = tuple(torch.randint(0, size, (1,)) for size in slice_mask.shape)
                random_index = random_indices[0].item(), random_indices[1].item(), random_indices[2].item(), random_indices[3].item()
                slice_mask[random_index] = True                
            reference_points_cams.append(reference_points_cam.to(slice_query[0].device))
            slice_masks.append(slice_mask.to(slice_query[0].device))

        for lid, layer in enumerate(self.layers):

            # Bug fix: wrong param names
            # output = layer(
            #     slice_query,
            #     key,
            #     value,
            #     *args,
            #     slice_pos=slice_pos,
            #     ref_2d=ref_2ds, #ref_cross_view,
            #     slice_h=slice_h,
            #     slice_w=slice_w,
            #     slice_z=slice_z,
            #     spatial_shapes=spatial_shapes,
            #     level_start_index=level_start_index,
            #     reference_points_cams=reference_points_cams,
            #     slice_masks=slice_masks,
            #     **kwargs)
            
            output = layer(
                slice_query,
                key,
                value,
                *args,
                tpv_pos=slice_pos,
                ref_2d=ref_2ds,
                tpv_h=slice_h,
                tpv_w=slice_w,
                tpv_z=slice_z,
                    spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                reference_points_cams=reference_points_cams,
                tpv_masks=slice_masks,
                **kwargs)
            
            slice_query = output


            if self.return_intermediate:
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output, ref_3ds

