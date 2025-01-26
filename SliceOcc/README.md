# SliceOcc: Indoor 3D Semantic Occupancy Prediction with Vertical Slice Representation

<div style="text-align: center;">
    <img src="https://i.imgs.ovh/2025/01/26/U7tX.md.jpeg" alt="Dialogue_Teaser" width=100% >
</div>

## 📋 Contents

1. [About](#🏠-about)
2. [Getting Started](#📚-getting-started)
3. [Model and Benchmark](#📦-model-and-benchmark)
4. [Citation](#🔗-citation)

## 🏠 About

3D semantic occupancy prediction is a crucial task in visual perception, as it requires the simultaneous comprehension of both scene geometry and semantics. It plays a crucial role in understanding 3D scenes and has great potential for various applications, such as robotic vision perception and autonomous driving.

In this paper, we present a new vertical slice representation that divides the scene along the vertical axis and projects spatial point features onto the nearest pair of parallel planes. To utilize these slice features, we propose SliceOcc, an RGB camera-based model specifically tailored for indoor 3D semantic occupancy prediction.

[demo](https://tai-wang.github.io/embodiedscan).

## 📚 Getting Started

### Installation

We test our codes under the following environment:

- Ubuntu 20.04
- NVIDIA Driver: 535.54.03
- CUDA 12.1
- Python 3.8.18
- PyTorch 1.11.0+cu113
- PyTorch3D 0.7.2

#### Steps:

1. **Clone the repository**:
    ```bash
    git clone https://github.com/NorthSummer/SliceOcc.git
    cd SliceOcc
    ```

2. **Create an environment and install PyTorch**:
    ```bash
    conda create -n embodiedscan python=3.8 -y
    conda activate embodiedscan
    conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch
    ```

3. **Install SliceOcc**:
    ```bash
    python install.py all  # Install all dependencies
    ```

### Data Preparation

Please refer to the [EmbodiedScan data preparation guide](https://github.com/OpenRobotLab/EmbodiedScan/tree/main/data) for downloading and organization.

## 📦 Model and Benchmark

### Training and Evaluation

We provide configs for different tasks [here](configs/) and you can run the train and test scripts in the [tools folder](tools/) for training and inference.

Example commands:

**Single GPU training**:
```bash
python tools/train.py configs/occupancy/mv-occ_8xb1_sliceformer-occ-80class.py --work-dir=work_dirs/sliceocc

# Multiple GPU training
python -m torch.distributed.launch --nproc_per_node=8 --master_port=25622 tools/train.py configs/occupancy/mv-occ_8xb1_sliceformer-occ-80class.py --launcher='pytorch' --work-dir=work_dirs/sliceocc
```

```bash
# Single GPU testing
python tools/test.py configs/occupancy/mv-occ_8xb1_sliceformer-occ-80class.py work_dirs/sliceocc/epoch_24.pth

```

### Multi-View Occupancy Prediction

| Method | Input | mIoU | 
|:------:|:-----:|:----:|
| SliceOcc | RGB-D | 15.46| 





