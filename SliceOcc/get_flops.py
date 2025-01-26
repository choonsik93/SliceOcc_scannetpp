from mmcv.cnn.utils.flops_counter import get_model_complexity_info
from functools import partial
from mmengine.config import Config
from embodiedscan.registry import MODELS, DATASETS

import warnings
from copy import deepcopy
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
from mmengine.config import Config
from mmengine.registry import init_default_scope
from mmengine.runner import load_checkpoint

from mmengine.analysis import get_model_complexity_info


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def init_model(config: Union[str, Path, Config],
               checkpoint: Optional[str] = None,
               device: str = 'cuda:0',
               cfg_options: Optional[dict] = None):
    """Initialize a model from config file, which could be a 3D detector or a
    3D segmentor.

    Args:
        config (str, :obj:`Path`, or :obj:`mmengine.Config`): Config file path,
            :obj:`Path`, or the config object.
        checkpoint (str, optional): Checkpoint path. If left as None, the model
            will not load any weights.
        device (str): Device to use.
        cfg_options (dict, optional): Options to override some settings in
            the used config.

    Returns:
        nn.Module: The constructed detector.
    """
    if isinstance(config, (str, Path)):
        config = Config.fromfile(config)
    elif not isinstance(config, Config):
        raise TypeError('config must be a filename or Config object, '
                        f'but got {type(config)}')
    if cfg_options is not None:
        config.merge_from_dict(cfg_options)

    config.model.train_cfg = None
    init_default_scope(config.get('default_scope', 'mmdet3d'))
    model = MODELS.build(config.model)

    if checkpoint is not None:
        checkpoint = load_checkpoint(model, checkpoint, map_location='cpu')
        # save the dataset_meta in the model for convenience
        model.dataset_meta = checkpoint['meta']['dataset_meta']

        test_dataset_cfg = deepcopy(config.test_dataloader.dataset)
        # lazy init. We only need the metainfo.
        test_dataset_cfg['lazy_init'] = True
        metainfo = DATASETS.build(test_dataset_cfg).metainfo
        cfg_palette = metainfo.get('palette', None)
        if cfg_palette is not None:
            model.dataset_meta['palette'] = cfg_palette
        else:
            if 'palette' not in model.dataset_meta:
                warnings.warn(
                    'palette does not exist, random is used by default. '
                    'You can also set the palette to customize.')
                model.dataset_meta['palette'] = 'random'

    model.cfg = config  # save the config in the model for convenience
    if device != 'cpu':
        torch.cuda.set_device(device)
    else:
        warnings.warn('Don\'t suggest using CPU device. '
                      'Some functions are not supported for now.')

    model.to(device)
    model.eval()
    return model

def init_dataset(config: Union[str, Path, Config], cfg_options: Optional[dict] = None):
    """Initialize a dataset from config file.

    Args:
        config (str, :obj:`Path`, or :obj:`mmengine.Config`): Config file path,
            :obj:`Path`, or the config object.
        cfg_options (dict, optional): Options to override some settings in
            the used config.

    Returns:
        Dataset: The constructed dataset.
    """
    if isinstance(config, (str, Path)):
        config = Config.fromfile(config)
    elif not isinstance(config, Config):
        raise TypeError('config must be a filename or Config object, '
                        f'but got {type(config)}')
    if cfg_options is not None:
        config.merge_from_dict(cfg_options)

    init_default_scope(config.get('default_scope', 'mmdet3d'))
    dataset = DATASETS.build(dict(type=config.dataset_type,
                                     data_root=config.data_root,
                                     ann_file='embodiedscan_infos_train.pkl',
                                     pipeline=config.train_pipeline,
                                     test_mode=False,
                                     filter_empty_gt=True,
                                     box_type_3d='Euler-Depth',
                                     metainfo=config.metainfo))
    return dataset

def input_constructor(dataset, *args, **kwargs):
    data = dataset[0]
    
    data_samples = data['data_samples']
    batch_img_metas = [data_samples.metainfo]
    #batch_img_metas = [data_samples]
    # for item in batch_img_metas[0]['depth2img']['extrinsic']:
    #     item = torch.tensor(item)
    #     print(item)
    # for item in batch_img_metas[0]['depth2img']['intrinsic']:
    #     item = torch.tensor(item)
    
    batch_img_metas[0]['depth2img']['extrinsic'] = list(torch.tensor(batch_img_metas[0]['depth2img']['extrinsic']).to(device))
    batch_img_metas[0]['depth2img']['intrinsic'] = list(torch.tensor(batch_img_metas[0]['depth2img']['intrinsic']).to(device))
    batch_img_metas[0]['depth2img']['origin'] = torch.tensor(batch_img_metas[0]['depth2img']['origin']).to(device)
    batch_img_metas[0]['axis_align_matrix'] = torch.tensor(batch_img_metas[0]['axis_align_matrix']).to(device)
    batch_img_metas[0]['cam2img'] = torch.tensor(batch_img_metas[0]['cam2img']).to(device)
    batch_img_metas[0]['scale_factor'] = torch.tensor(batch_img_metas[0]['scale_factor']).to(device)
    batch_img_metas[0]['sample_idx'] = torch.tensor(batch_img_metas[0]['sample_idx']).to(device)
    batch_img_metas[0]['ori_shape'] = torch.tensor(batch_img_metas[0]['ori_shape']).to(device)
    batch_img_metas[0]['img_shape'] = torch.tensor(batch_img_metas[0]['img_shape']).to(device)
    
    batch_img_metas[0].pop('box_type_3d')
        
    inputs = data['inputs']
    for key in inputs.keys():
        inputs[key] = torch.tensor(inputs[key], dtype=torch.float32).to(device)
    #print(inputs)
    return inputs, batch_img_metas

def build_model_dm():
    config_path = '/mnt/data/ljn/code/EmbodiedScan/configs/occupancy/mv-occ_8xb1_sliceformer-occ-80class.py'
    #config_path = '/mnt/data/ljn/code/EmbodiedScan/configs/occupancy/mv-occ_8xb1_embodiedscan-occ-80class.py'
    model = init_model(config_path, None, device=device)
    dataset = init_dataset(config_path)
    input_data = input_constructor(dataset)
    analysis_results = get_model_complexity_info(model, input_shape=None, inputs=input_data)
    print("Model Flops:{}".format(analysis_results['flops_str']))
    print("Model Parameters:{}".format(analysis_results['params_str']))

if __name__ == '__main__':
    build_model_dm()