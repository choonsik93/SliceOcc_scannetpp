# SliceOcc: Indoor 3D Semantic Occupancy Prediction with Vertical Slice Representation

<div style="text-align: center;">
    <img src="https://i.imgs.ovh/2025/01/26/U7tX.jpeg" alt="Dialogue_Teaser" width=100% >
</div>

## 📋 Contents

1. [Getting Started](#📚-getting-started)
2. [Model and Benchmark](#📦-model-and-benchmark)

## 🏠 About

Scannetpp dataset을 이용가능하도록 한 Sliceocc source 코드 확장

## 📚 Getting Started

### Installation

#### Steps:

1. **Clone the repository**:
    ```bash
    git clone https://github.com/choonsik93/SliceOcc_scannetpp.git
    cd SliceOcc_scannetpp
    ```

2. **Create an Docker image**:
    ```bash
    make build-image
    ```

3. **Run the installed Docker image**:
    ```bash
    export SCANNET_PATH=/path/to/your/scannetpp/data/folder && make run
    ```

### Data Preparation

#### Steps:

1. **Create the scannetpp occupancy files**:
    ```bash
    python -m embodiedscan.converter.generate_scannetpp_occupancy
    ```

2. **Create the scannetpp train and valid infos**:
    ```bash
    python -m embodiedscan.converter.generate_scannetpp_info
    ```

## 📦 Model and Benchmark

### Training and Evaluation

We provide configs for different tasks [here](configs/) and you can run the train and test scripts in the [tools folder](tools/) for training and inference.

Example commands:

**Single GPU training**:
```bash
tools/train.py configs/occupancy/mv-occ_8xb1_sliceformer-occ-80class.py --work-dir=work_dirs/sliceocc

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





