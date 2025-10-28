import torchsparse
import torchsparse.nn as spnn
from torch import nn
from torchsparse import SparseTensor
from torchsparse.utils.quantize import sparse_quantize

import numpy as np
import torch
import torchsparse
import torchsparse.nn.functional as spf
from torchsparse import SparseTensor
from torchsparse.nn.functional.devoxelize import calc_ti_weights
from torchsparse.nn.utils import *
from torchsparse.utils import *

import torch_scatter
from typing import Union, Tuple
from embodiedscan.registry import MODELS
#__all__ = ["initial_voxelize", "point_to_voxel", "voxel_to_point", "PointTensor"]

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


class PointTensor(SparseTensor):
    def __init__(
        self,
        feats: torch.Tensor,
        coords: torch.Tensor,
        stride: Union[int, Tuple[int, ...]] = 1,
    ):
        super().__init__(feats=feats, coords=coords, stride=stride)
        self._caches.idx_query = dict()
        self._caches.idx_query_devox = dict()
        self._caches.weights_devox = dict()


def sphashquery(query, target, kernel_size=1):
    hashmap_keys = torch.zeros(
        2 * target.shape[0], dtype=torch.int64, device=target.device
    )
    hashmap_vals = torch.zeros(
        2 * target.shape[0], dtype=torch.int32, device=target.device
    )
    hashmap = torchsparse.backend.GPUHashTable(hashmap_keys, hashmap_vals)
    hashmap.insert_coords(target[:, [1, 2, 3, 0]])
    kernel_size = make_ntuple(kernel_size, 3)
    kernel_volume = np.prod(kernel_size)
    #print(kernel_volume)
    kernel_size = make_tensor(kernel_size, device=target.device, dtype=torch.int32)
    stride = make_tensor((1, 1, 1), device=target.device, dtype=torch.int32)
    #print(query.device, kernel_size.device, stride.device)
    results = (
        hashmap.lookup_coords(
            query[:, [1, 2, 3, 0]].detach().cpu(), kernel_size.detach().cpu(), stride.detach().cpu(), kernel_volume
        )
        #- 1
    )[: query.shape[0]]
    return results

# z: PointTensor
# return: SparseTensor
def initial_voxelize(z, init_res, after_res):
    new_float_coord = torch.cat(
        [z.C[:, 0].view(-1, 1), (z.C[:, 1:] * init_res) / after_res], 1
    )
    # optimization TBD: init_res = after_res
    new_int_coord = torch.floor(new_float_coord).int()
    sparse_coord = torch.unique(new_int_coord, dim=0)
    #print(torch.min(sparse_coord), torch.max(sparse_coord))
    idx_query = sphashquery(new_int_coord, sparse_coord).reshape(-1) 
    #print(z.F.shape, idx_query.shape, z.F.device, idx_query.device) torch.Size([100000, 3]) torch.Size([100000]) cuda:0 cuda:0
    print(idx_query.device, z.F.device)
    print(torch.max(idx_query), torch.min(idx_query))
    #print(torch.max(z.F[:,0]), torch.max(z.F[:,1]), torch.max(z.F[:,2]), torch.min(z.F[:,0]), torch.min(z.F[:,1]), torch.min(z.F[:,2]))
    sparse_feat = torch_scatter.scatter_mean(z.F.detach().cpu(), idx_query.long(), dim=0)
    new_tensor = SparseTensor(sparse_feat, sparse_coord, 1)
    z._caches.idx_query[z.s] = idx_query
    z.C = new_float_coord
    return new_tensor

    


# x: SparseTensor, z: PointTensor
# return: SparseTensor
def point_to_voxel(x, z):
    if z._caches.idx_query.get(x.s) is None:
        # Note: x.C has a smaller range after downsampling.
        new_int_coord = torch.cat(
            [
                z.C[:, 0].int().view(-1, 1),
                torch.floor(z.C[:, 1:] / x.s[0]).int(),
            ],
            1,
        )
        idx_query = sphashquery(new_int_coord, x.C)
        z._caches.idx_query[x.s] = idx_query
    else:
        idx_query = z._caches.idx_query[x.s]
    # Haotian: This impl. is not elegant
    idx_query = idx_query.clamp_(0)
    sparse_feat = torch_scatter.scatter_mean(z.F, idx_query.long(), dim=0)
    new_tensor = SparseTensor(sparse_feat, x.C, x.s)
    new_tensor._caches = x._caches

    return new_tensor


# x: SparseTensor, z: PointTensor
# return: PointTensor
def voxel_to_point(x, z, nearest=False):
    if (
        z._caches.idx_query_devox.get(x.s) is None
        or z._caches.weights_devox.get(x.s) is None
    ):
        point_coords_float = torch.cat(
            [z.C[:, 0].int().view(-1, 1), z.C[:, 1:] / x.s[0]],
            1,
        )
        point_coords_int = torch.floor(point_coords_float).int()
        idx_query = sphashquery(point_coords_int, x.C, kernel_size=2)
        weights = calc_ti_weights(point_coords_float[:, 1:], idx_query, scale=1)

        if nearest:
            weights[:, 1:] = 0.0
            idx_query[:, 1:] = -1
        new_feat = spf.spdevoxelize(x.F, idx_query, weights)
        new_tensor = PointTensor(new_feat, z.C)
        new_tensor._caches = z._caches
        new_tensor._caches.idx_query_devox[x.s] = idx_query
        new_tensor._caches.weights_devox[x.s] = weights
        z._caches.idx_query_devox[x.s] = idx_query
        z._caches.weights_devox[x.s] = weights

    else:
        new_feat = spf.spdevoxelize(
            x.F, z._caches.idx_query_devox.get(x.s), z._caches.weights_devox.get(x.s)
        )
        new_tensor = PointTensor(new_feat, z.C)
        new_tensor._caches = z._caches

    return new_tensor


class BasicConvolutionBlock(nn.Module):

    def __init__(self, inc, outc, ks=3, stride=1, dilation=1):
        super().__init__()
        self.net = nn.Sequential(
            spnn.Conv3d(inc,
                        outc,
                        kernel_size=ks,
                        dilation=dilation,
                        stride=stride),
            spnn.BatchNorm(outc),
            spnn.ReLU(True),
        )

    def forward(self, x):
        out = self.net(x)
        return out


class BasicDeconvolutionBlock(nn.Module):

    def __init__(self, inc, outc, ks=3, stride=1):
        super().__init__()
        self.net = nn.Sequential(
            spnn.Conv3d(inc,
                        outc,
                        kernel_size=ks,
                        stride=stride,
                        transposed=True),
            spnn.BatchNorm(outc),
            spnn.ReLU(True),
        )

    def forward(self, x):
        return self.net(x)


class ResidualBlock(nn.Module):

    def __init__(self, inc, outc, ks=3, stride=1, dilation=1):
        super().__init__()
        self.net = nn.Sequential(
            spnn.Conv3d(inc,
                        outc,
                        kernel_size=ks,
                        dilation=dilation,
                        stride=stride),
            spnn.BatchNorm(outc),
            spnn.ReLU(True),
            spnn.Conv3d(outc, outc, kernel_size=ks, dilation=dilation,
                        stride=1),
            spnn.BatchNorm(outc),
        )

        if inc == outc and stride == 1:
            self.downsample = nn.Identity()
        else:
            self.downsample = nn.Sequential(
                spnn.Conv3d(inc, outc, kernel_size=1, dilation=1,
                            stride=stride),
                spnn.BatchNorm(outc),
            )

        self.relu = spnn.ReLU(True)

    def forward(self, x):
        out = self.relu(self.net(x) + self.downsample(x))
        return out

@MODELS.register_module()
class SPVCNN(nn.Module):

    def __init__(self, **kwargs):
        super().__init__()

        cr = kwargs.get('cr', 1.0)
        cs = [32, 32, 64, 128, 256, 256, 128, 96, 256]
        cs = [int(cr * x) for x in cs]

        if 'pres' in kwargs and 'vres' in kwargs:
            self.pres = kwargs['pres']
            self.vres = kwargs['vres']

        self.stem = nn.Sequential(
            spnn.Conv3d(4, cs[0], kernel_size=3, stride=1),
            spnn.BatchNorm(cs[0]), spnn.ReLU(True),
            spnn.Conv3d(cs[0], cs[0], kernel_size=3, stride=1),
            spnn.BatchNorm(cs[0]), spnn.ReLU(True))

        self.stage1 = nn.Sequential(
            BasicConvolutionBlock(cs[0], cs[0], ks=2, stride=2, dilation=1),
            ResidualBlock(cs[0], cs[1], ks=3, stride=1, dilation=1),
            ResidualBlock(cs[1], cs[1], ks=3, stride=1, dilation=1),
        )

        self.stage2 = nn.Sequential(
            BasicConvolutionBlock(cs[1], cs[1], ks=2, stride=2, dilation=1),
            ResidualBlock(cs[1], cs[2], ks=3, stride=1, dilation=1),
            ResidualBlock(cs[2], cs[2], ks=3, stride=1, dilation=1),
        )

        self.stage3 = nn.Sequential(
            BasicConvolutionBlock(cs[2], cs[2], ks=2, stride=2, dilation=1),
            ResidualBlock(cs[2], cs[3], ks=3, stride=1, dilation=1),
            ResidualBlock(cs[3], cs[3], ks=3, stride=1, dilation=1),
        )

        self.stage4 = nn.Sequential(
            BasicConvolutionBlock(cs[3], cs[3], ks=2, stride=2, dilation=1),
            ResidualBlock(cs[3], cs[4], ks=3, stride=1, dilation=1),
            ResidualBlock(cs[4], cs[4], ks=3, stride=1, dilation=1),
        )

        self.up1 = nn.ModuleList([
            BasicDeconvolutionBlock(cs[4], cs[5], ks=2, stride=2),
            nn.Sequential(
                ResidualBlock(cs[5] + cs[3], cs[5], ks=3, stride=1, dilation=1),
                ResidualBlock(cs[5], cs[5], ks=3, stride=1, dilation=1),
            )
        ])

        self.up2 = nn.ModuleList([
            BasicDeconvolutionBlock(cs[5], cs[6], ks=2, stride=2),
            nn.Sequential(
                ResidualBlock(cs[6] + cs[2], cs[6], ks=3, stride=1, dilation=1),
                ResidualBlock(cs[6], cs[6], ks=3, stride=1, dilation=1),
            )
        ])

        self.up3 = nn.ModuleList([
            BasicDeconvolutionBlock(cs[6], cs[7], ks=2, stride=2),
            nn.Sequential(
                ResidualBlock(cs[7] + cs[1], cs[7], ks=3, stride=1, dilation=1),
                ResidualBlock(cs[7], cs[7], ks=3, stride=1, dilation=1),
            )
        ])

        self.up4 = nn.ModuleList([
            BasicDeconvolutionBlock(cs[7], cs[8], ks=2, stride=2),
            nn.Sequential(
                ResidualBlock(cs[8], cs[8], ks=3, stride=1, dilation=1),
                ResidualBlock(cs[8], cs[8], ks=3, stride=1, dilation=1),
            )
        ])

        self.classifier = nn.Sequential(nn.Linear(cs[8], 256))

        self.point_transforms = nn.ModuleList([
            nn.Sequential(
                nn.Linear(cs[0], cs[4]),
                nn.BatchNorm1d(cs[4]),
                nn.ReLU(True),
            ),
            nn.Sequential(
                nn.Linear(cs[4], cs[6]),
                nn.BatchNorm1d(cs[6]),
                nn.ReLU(True),
            ),
            nn.Sequential(
                nn.Linear(cs[6], cs[8]),
                nn.BatchNorm1d(cs[8]),
                nn.ReLU(True),
            )
        ])

        self.weight_initialization()
        self.dropout = nn.Dropout(0.3, True)

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, voxel_size):
        # x: SparseTensor z: PointTensor
                

        pc_ = np.round(x[:, :3].detach().cpu().numpy() / voxel_size).astype(np.int32)
        pc_ -= pc_.min(0, keepdims=1)
        
        #print(pc_.shape)      
        _, inds, inverse_map = sparse_quantize(pc_,
                                               return_index=True,
                                               return_inverse=True)
        #print(pc_.shape, inds.shape)
        pc_ = pc_[inds]
        x = x[inds]
        n = x.shape[0]
        intensity = torch.ones((n, 1), dtype=x.dtype, device=x.device)
        x = torch.cat((x, intensity), dim=1)

        pc_ = torch.IntTensor(pc_).to(x.device)
        bs = torch.zeros((n, 1), dtype=pc_.dtype, device=pc_.device)
        pc_ = torch.cat((bs, pc_), dim=1)
        y = SparseTensor(x, pc_)

        z = PointTensor(torch.Tensor(y.F).to(x.device), torch.Tensor(y.C).float().to(x.device))

        #x0 = initial_voxelize(z, self.pres, self.vres)
        
        x0 = self.stem(y)
        #z0 = voxel_to_point(x0, z, nearest=False)
        #z0.F = z0.F
        #print(x0.F.shape, x0.C.shape)
        
        #x1 = point_to_voxel(x0, z0)
        x1 = self.stage1(x0)
        x2 = self.stage2(x1)
        x3 = self.stage3(x2)
        #x4 = self.stage4(x3)
        #z1 = voxel_to_point(x4, z0)
        #z1.F = z1.F + self.point_transforms[0](z0.F)

        #y1 = point_to_voxel(x4, z1)
        #y1.F = self.dropout(y1.F)
        '''
        y1 = self.up1[0](x4)
        y1 = torchsparse.cat([y1, x3])
        y1 = self.up1[1](y1)

        y2 = self.up2[0](y1)
        y2 = torchsparse.cat([y2, x2])
        y2 = self.up2[1](y2)
        z2 = voxel_to_point(y2, z1)
        z2.F = z2.F + self.point_transforms[1](z1.F)

        y3 = point_to_voxel(y2, z2)
        y3.F = self.dropout(y3.F)
        y3 = self.up3[0](y3)
        y3 = torchsparse.cat([y3, x1])
        y3 = self.up3[1](y3)

        y4 = self.up4[0](y3)
        y4 = torchsparse.cat([y4, x0])
        y4 = self.up4[1](y4)
        z3 = voxel_to_point(y4, z2)
        z3.F = z3.F + self.point_transforms[2](z2.F)
        '''

        y1 = self.up2[0](x3)
        y1 = torchsparse.cat([y1, x2])
        y1 = self.up2[1](y1)

        y2 = self.up3[0](y1)
        y2 = torchsparse.cat([y2, x1])
        y2 = self.up3[1](y2)
        #z2 = voxel_to_point(y2, z1)
        #z2.F = z2.F + self.point_transforms[1](z1.F)

        #y3 = point_to_voxel(y2, z2)
        #y3.F = self.dropout(y3.F)
        y3 = self.up4[0](y2)
        #y3 = torchsparse.cat([y2, x0])
        y3 = self.up4[1](y3)
        y3.spatial_range = (40,40,16)

        #out = self.classifier(y3.F)
        #print(y3.C.shape, y3.F.shape)
        #print(torch.max(y3.C[:,0]), torch.max(y3.C[:,1]), torch.max(y3.C[:,2]), torch.max(y3.C[:,3]))
        out = y3.dense().permute(3,0,1,2).unsqueeze(0)
        #print(out.shape)
        return out