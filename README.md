# 基于背景干扰物挖掘的目标跟踪算法研究与实现

## 项目概述

本项目基于 SiamRPN++ 跟踪框架，融合 CBAM 注意力机制与 HOG+CNN 多特征，
实现对背景干扰物的主动挖掘、风险分级及跟踪策略自适应调整。

## 核心模块

| 模块 | 说明 |
|------|------|
| `models/` | 模型定义（SiamRPN++ backbone、CBAM、RPN head） |
| `core/` | 核心算法逻辑（跟踪器、干扰物管理、融合优化） |
| `data/` | 数据集加载与预处理 |
| `utils/` | 工具函数（可视化、日志、指标计算、特征提取） |
| `configs/` | 配置文件（YAML） |
| `scripts/` | 训练/评估/演示入口脚本 |

## 环境要求

- Python 3.9
- PyTorch 2.8, torchvision 0.23(本人使用环境，根据情况调整)
- OpenCV, scikit-image, scipy, pyyaml


## 项目结构

```
distractor_tracking/
├── configs/                # 配置文件
│   └── default.yaml
├── models/                 # 模型
│   ├── backbone.py         # 骨干网络 (ResNet50 / MobileNetV3)
│   ├── cbam.py             # CBAM 注意力模块
│   ├── rpn.py              # Region Proposal Network
│   ├── neck.py             # 特征融合颈部网络
│   └── siamrpn.py          # SiamRPN++ 完整模型
├── core/                   # 核心算法
│   ├── tracker.py          # 跟踪主控器
│   ├── distractor.py       # 干扰物管理器
│   └── fusion.py           # 融合策略 + 干扰物加权损失
├── data/                   # 数据处理
│   ├── datasets.py         # 数据集加载器
│   ├── anchor.py           # Anchor 网格生成 + IoU 标签分配
│   ├── train_dataset.py    # 训练 Dataset（帧对采样 + 标签生成）
│   └── transforms.py       # 数据增强与预处理
├── utils/                  # 工具
│   ├── logger.py           # 日志
│   ├── visualizer.py       # 可视化工具
│   ├── metrics.py          # 评估指标
│   └── features.py         # HOG 特征提取工具
├── scripts/                # 训练脚本
│   ├── check_train_data.py # 训练前健全性检查
│   ├── train.py            # 训练
│   ├── evaluate.py         # 评估
│   └── demo.py             # 单视频演示
├── requirements.txt
└── README.md
```

## 1. 环境准备

```bash
# Python >= 3.8, 建议 conda 管理
conda create -n tracker python=3.9 -y
conda activate tracker

# PyTorch (根据你的 CUDA 版本选择)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 其他依赖
pip install opencv-python-headless numpy matplotlib pyyaml scikit-image
```

## 2. 数据集准备

在项目根目录下 `data/` 文件夹，在此文件夹下放置数据集

## 3. 配置检查

确认 `configs/default.yaml` 中的数据集路径正确：

```yaml
datasets:
  otb100:
    root: "data/OTB100/"   # ← 改成你的实际路径
    type: "otb"
```

GPU 设置：

```yaml
system:
  device: "cuda:0"         # 无 GPU 改为 "cpu"
  num_workers: 4           # Windows 可能需要改为 0
```

## 4. 执行顺序

### Step 1: 启动训练

```bash
# 首次训练
python scripts/train.py --config configs/default.yaml

# 断点续训
python scripts/train.py --config configs/default.yaml --resume checkpoints/latest.pth
```

**检查点保存位置**: `checkpoints/latest.pth` + 每 10 epoch 一份 `epoch_N.pth`

### Step 2: 评估

```bash
python scripts/evaluate.py --config configs/default.yaml --checkpoint .\checkpoints\best.pth --dataset got10k_val  
```

计算成功率、精确度、FPS。
