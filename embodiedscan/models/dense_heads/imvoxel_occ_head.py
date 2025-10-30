# Copyright (c) OpenRobotLab. All rights reserved.

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import build_conv_layer
from mmengine.model import BaseModule
from torch import Tensor

from embodiedscan.models.losses.occ_loss import (geo_scal_loss,
                                                 occ_multiscale_supervision,
                                                 sem_scal_loss)
from embodiedscan.registry import MODELS
from embodiedscan.utils.typing_config import SampleList


@MODELS.register_module()
class ImVoxelOccHead(BaseModule):
    """Occupancy prediction head compatible with ImVoxelNeck outputs.

    Args:
        num_classes (int): Number of categories. Defaults to 21.
        volume_h (int): Size along h of the 3D volume. Defaults to 40.
        volume_w (int): Size along w of the 3D volume. Defaults to 40.
        volume_z (int): Size along z of the 3D volume. Defaults to 16.
        in_channels (int): Input channels. Defaults to 128.
        use_semantic (bool): Whether to use semantic predictions.
            Defaults to True.
    """

    def __init__(self,
                 *args,
                 num_classes=21,
                 volume_h=40,
                 volume_w=40,
                 volume_z=16,
                 in_channels=128,
                 use_semantic=True,
                 up_ratio=-1,
                 **kwargs):
        super(ImVoxelOccHead, self).__init__()
        self.num_classes = num_classes
        self.volume_h = volume_h
        self.volume_w = volume_w
        self.volume_z = volume_z
        self.in_channels = in_channels
        self.use_semantic = use_semantic
        self.up_ratio = up_ratio

        self._init_layers()

    def _init_layers(self):
        conv_cfg = dict(type='Conv3d', bias=False)
        self.occ = nn.ModuleList()
        for i in range(len(self.in_channels)):
            if self.use_semantic:
                occ = build_conv_layer(conv_cfg,
                                       in_channels=self.in_channels[i],
                                       out_channels=self.num_classes,
                                       kernel_size=1,
                                       stride=1,
                                       padding=0)
                self.occ.append(occ)
            else:
                occ = build_conv_layer(conv_cfg,
                                       in_channels=self.in_channels[i],
                                       out_channels=1,
                                       kernel_size=1,
                                       stride=1,
                                       padding=0)
                self.occ.append(occ)

        if self.up_ratio != -1:
            self.up_occ = nn.ModuleList()
            deconv_cfg = dict(type='ConvTranspose3d', bias=False)
            deconv_occ = build_conv_layer(
                deconv_cfg,
                in_channels=self.in_channels[0],
                out_channels=self.in_channels[0],
                kernel_size=self.up_ratio,
                stride=self.up_ratio,
                padding=0,
            )
            self.up_occ.append(deconv_occ)
            if self.use_semantic:
                occ = build_conv_layer(conv_cfg,
                                       in_channels=self.in_channels[0],
                                       out_channels=self.num_classes,
                                       kernel_size=1,
                                       stride=1,
                                       padding=0)
                self.up_occ.append(occ)
            else:
                occ = build_conv_layer(conv_cfg,
                                       in_channels=self.in_channels[0],
                                       out_channels=1,
                                       kernel_size=1,
                                       stride=1,
                                       padding=0)
                self.up_occ.append(occ)

        # for i in range(len(self.in_channels - 1)):
        #     deconv_cfg = dict(type='ConvTranspose3d', bias=False)
        #     up_stride = self.volume_h[i + 1] // self.volume_h[i]
        #     deconv_occ = build_conv_layer(
        #         deconv_cfg,
        #         in_channels=self.in_channels[-1],
        #         out_channels=self.in_channels[-1],
        #         kernel_size=up_stride,
        #         stride=up_stride,
        #         padding=0,
        #     )
        #     print(deconv_occ)


    def forward(self, mlvl_feats, input_metas):
        """Forward function.

        Args:
            mlvl_feats (list[Tensor]): Multi-level features.
            input_metas (list[dict]): Input meta infos.

        Returns:
            list[Tensor]: Occupancy predicted maps.
        """
        occ_preds = []
        for i in range(len(mlvl_feats)):
            occ_pred = self.occ[i](mlvl_feats[i])
            occ_preds.append(occ_pred)

        if self.up_ratio != -1:
            up_occ_feat = self.up_occ[0](mlvl_feats[0])
            occ_pred = self.up_occ[1](up_occ_feat)
            occ_preds = [occ_pred] + occ_preds

        return occ_preds

    def predict(self, x: Tuple[Tensor], batch_data_samples: SampleList):
        """Predict/Inference function.

        Args:
            x (Tuple[Tensor]): Multi-level features.
            batch_data_samples (`SampleList`): Batch of data samples.

        Returns:
            Tensor: Occupancy predictions.
        """
        batch_input_metas = [
            data_samples.metainfo for data_samples in batch_data_samples
        ]
        pred = self.forward(x, batch_input_metas)[0]
        if self.use_semantic:
            _, pred_occ = torch.max(torch.softmax(pred, dim=1), dim=1)
        else:
            pred_occ = torch.sigmoid(pred[:, 0])
        return pred_occ

    def loss(self, x: Tuple[Tensor], batch_data_samples: SampleList):
        """Loss function..

        Args:
            x (Tuple[Tensor]): Multi-level features.
            batch_data_samples (`SampleList`): Batch of data samples.

        Returns:
            Dict: Multi-scale occupancy prediction losses.
        """
        batch_input_metas = [
            data_samples.metainfo for data_samples in batch_data_samples
        ]  
        occ_preds = self.forward(x, batch_input_metas)

        gt_occupancy = [
            data_samples.gt_occupancy for data_samples in batch_data_samples
        ]  # B * N * 4, xyz+label, TODO: put data in load process
        #print(len(gt_occupancy)) 1

        gt_occupancy_masks = None
        if 'gt_occupancy_masks' in batch_data_samples[0]:
            gt_occupancy_masks = [
                data_samples.gt_occupancy_masks
                for data_samples in batch_data_samples
            ]

        if not self.use_semantic:
            loss_dict = {}
            for i in range(len(occ_preds)):
                pred = occ_preds[i][:, 0]
                ratio = self.volume_w // pred.shape[1]
                # downsample occ_masks accordingly
                if gt_occupancy_masks is not None:
                    pooling = nn.MaxPool3d(ratio, stride=ratio)
                    pooled_masks = []
                    for mask in gt_occupancy_masks:
                        float_mask = mask.float()[None]
                        pooled_masks.append(pooling(float_mask)[0].bool())
                else:
                    pooled_masks = gt_occupancy_masks
                gt = occ_multiscale_supervision(gt_occupancy, ratio,
                                                occ_preds[i].shape,
                                                pooled_masks)
                loss_occ_i = (F.binary_cross_entropy_with_logits(pred, gt) +
                              geo_scal_loss(pred, gt.long(), semantic=False))
                loss_occ_i = loss_occ_i * ((0.5)**i)
                loss_dict['loss_occ_{}'.format(i)] = loss_occ_i

        else:
            pred = occ_preds
            criterion = nn.CrossEntropyLoss(ignore_index=255, reduction='mean')
            loss_dict = {}

            for i in range(len(occ_preds)):
                pred = occ_preds[i]
                ratio = self.volume_w // pred.shape[2]
                # downsample occ_masks accordingly
                if gt_occupancy_masks is not None:
                    # a little hack: use maxpool to achieve downsample
                    pooling = nn.MaxPool3d(ratio, stride=ratio)
                    pooled_masks = []
                    for mask in gt_occupancy_masks:
                        float_mask = mask.float()[None]
                        pooled_masks.append(pooling(float_mask)[0].bool())
                else:
                    pooled_masks = gt_occupancy_masks
                gt = occ_multiscale_supervision(gt_occupancy, ratio,
                                                occ_preds[i].shape,
                                                pooled_masks)
                loss_occ_i = (criterion(pred, gt.long()) +
                              sem_scal_loss(pred, gt.long()) +
                              geo_scal_loss(pred, gt.long()))
                loss_occ_i = loss_occ_i * ((0.5)**i)
                loss_dict['loss_occ_{}'.format(i)] = loss_occ_i
                
                # B = gt.size(0)
                # has_valid = (gt != 255).reshape(B, -1).any(dim=1)
                # valid_batch_idx = has_valid.nonzero(as_tuple=False).squeeze(1)

                # if valid_batch_idx.numel() == 0:
                #     loss_occ_i = torch.Tensor([0.]).to(pred.device)
                # else:
                #     pred = pred.index_select(0, valid_batch_idx)  # (Nb, C, W, H, D)
                #     gt = gt.index_select(0, valid_batch_idx)    # (Nb, W, H, D)
                #     loss_occ_i = (criterion(pred, gt.long()) +
                #                 sem_scal_loss(pred, gt.long()) +
                #                 geo_scal_loss(pred, gt.long()))
                #     loss_occ_i = loss_occ_i * ((0.5)**i)
                # loss_dict['loss_occ_{}'.format(i)] = loss_occ_i

        return loss_dict
