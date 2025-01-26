
<br>
<p align="center">
<h1 align="center"><strong>SliceOcc: Indoor 3D Semantic Occupancy Prediction with Vertical Slice
Representation</strong></h1>



<div id="top" align="center">



<!-- <div style="text-align: center;">
    <img src="assets/demo_fig.png" alt="Dialogue_Teaser" width=100% >
</div> -->

[![demo](assets/demo_fig.png "demo")](https://tai-wang.github.io/embodiedscan)

<!-- contents with emoji -->

## 📋 Contents

1. [About](#-about)
2. [Getting Started](#-getting-started)
3. [Citation](#-citation)

## 🏠 About

<!-- ![Teaser](assets/teaser.jpg) -->

<div style="text-align: center;">
    <img src="assets/teaser.png" alt="Dialogue_Teaser" width=100% >
</div>
3D semantic occupancy prediction is a crucial task in visual perception, as it requires the simultaneous comprehension of both scene geometry and semantics. It plays a crucial role in understanding 3D scenes and has great potential for various applications, such as robotic vision perception and autonomous driving. Many existing works utilize planarbased representations such as Bird’s Eye View (BEV) and Tri- Perspective View (TPV). These representations aim to simplify the complexity of 3D scenes while preserving essential object information, thereby facilitating efficient scene representation. However, in dense indoor environments with prevalent occlusions, directly applying these planar-based methods often leads to difficulties in capturing global semantic occupancy, ultimately degrading model performance. In this paper, we present a new vertical slice representation that divides the scene along the vertical axis and projects spatial point features onto the nearest pair of parallel planes. To utilize these slice features, we propose SliceOcc, an RGB camera-based model specifically tailored for indoor 3D semantic occupancy prediction. SliceOcc utilizes pairs of slice queries and cross-attention mechanisms to extract planar features from input images. These local planar features are then fused to form a global scene representation, which is employed for indoor occupancy prediction. Experimental results on the EmbodiedScan dataset demonstrate that SliceOcc achieves a mIoU of 15.45% across 81 indoor categories, setting a new state-of-the-art performance among RGB camerabased
models for indoor 3D semantic occupancy prediction.



## 📚 Getting Started

### Installation

We test our codes under the following environment:

- Ubuntu 20.04
- NVIDIA Driver: 535.54.03
- CUDA 12.1
- Python 3.8.18
- PyTorch 1.11.0+cu113
- PyTorch3D 0.7.2

1. Clone this repository.

```bash
git clone https://github.com/NorthSummer/SliceOcc.git
cd SliceOcc
```

2. Create an environment and install PyTorch.

```bash
conda create -n embodiedscan python=3.8 -y  # pytorch3d needs python>3.7
conda activate embodiedscan
# Install PyTorch, for example, install PyTorch 1.11.0 for CUDA 11.3
# For more information, please refer to https://pytorch.org/get-started/locally/
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch
```

3. Install SliceOcc.

```bash
# We plan to make EmbodiedScan easier to install by "pip install EmbodiedScan".
# Please stay tuned for the future official release.
# Make sure you are under ./EmbodiedScan/
# This script will install the dependencies and EmbodiedScan package automatically.
# use [python install.py run] to install only the execution dependencies
# use [python install.py visual] to install only the visualization dependencies
python install.py all  # install all the dependencies
```



### Data Preparation

Please refer to the [EmbodiedScan data preparation guide](https://github.com/OpenRobotLab/EmbodiedScan/tree/main/data) for downloading and organization.



## 📦 Model and Benchmark


### Training and Evaluation

We provide configs for different tasks [here](configs/) and you can run the train and test script in the [tools folder](tools/) for training and inference.
For example, to train a multi-view 3D detection model with pytorch, just run:

```bash
# Single GPU training
python tools/train.py configs/occupancy/mv-occ_8xb1_sliceformer-occ-80class.py --work-dir=work_dirs/sliceocc

# Multiple GPU training
python -m torch.distributed.launch --nproc_per_node=8 --master_port=25622 tools/train.py configs/occupancy/mv-occ_8xb1_sliceformer-occ-80class.py --launcher='pytorch' --work-dir=work_dirs/sliceocc
```



```bash
# Single GPU testing
python tools/test.py configs/occupancy/mv-occ_8xb1_sliceformer-occ-80class.py work_dirs/sliceocc/epoch_24.pth





#### Multi-View Occupancy Prediction

| Method | Input | mIoU | 
|:------:|:-----:|:----:|
| SliceOcc | RGB-D | 15.46| 





## 🔗 Citation

If you find our work helpful, please cite:



Please kindly cite the original datasets involved in our work. BibTex entries are provided below.

<details><summary>Dataset BibTex</summary>

```bibtex
@inproceedings{wang2023embodiedscan,
    title={EmbodiedScan: A Holistic Multi-Modal 3D Perception Suite Towards Embodied AI},
    author={Wang, Tai and Mao, Xiaohan and Zhu, Chenming and Xu, Runsen and Lyu, Ruiyuan and Li, Peisen and Chen, Xiao and Zhang, Wenwei and Chen, Kai and Xue, Tianfan and Liu, Xihui and Lu, Cewu and Lin, Dahua and Pang, Jiangmiao},
    year={2024},
    booktitle={IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
}
```

```BibTex
@inproceedings{dai2017scannet,
  title={ScanNet: Richly-annotated 3D Reconstructions of Indoor Scenes},
  author={Dai, Angela and Chang, Angel X. and Savva, Manolis and Halber, Maciej and Funkhouser, Thomas and Nie{\ss}ner, Matthias},
  booktitle = {Proceedings IEEE Computer Vision and Pattern Recognition (CVPR)},
  year = {2017}
}
```

```BibTex
@inproceedings{Wald2019RIO,
  title={RIO: 3D Object Instance Re-Localization in Changing Indoor Environments},
  author={Johanna Wald, Armen Avetisyan, Nassir Navab, Federico Tombari, Matthias Niessner},
  booktitle={Proceedings IEEE International Conference on Computer Vision (ICCV)},
  year = {2019}
}
```

```BibTex
@article{Matterport3D,
  title={{Matterport3D}: Learning from {RGB-D} Data in Indoor Environments},
  author={Chang, Angel and Dai, Angela and Funkhouser, Thomas and Halber, Maciej and Niessner, Matthias and Savva, Manolis and Song, Shuran and Zeng, Andy and Zhang, Yinda},
  journal={International Conference on 3D Vision (3DV)},
  year={2017}
}
```


