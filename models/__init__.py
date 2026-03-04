"""
models 包 — 网络模型定义
===========================
包含 SiamRPN++ 跟踪框架的全部网络组件：
  - backbone: 骨干特征提取网络 (ResNet50 / MobileNetV3)
  - cbam:     CBAM 通道-空间双注意力模块
  - neck:     特征融合调整层 (Adjust Layer)
  - rpn:      区域候选网络 (分类头 + 回归头)
  - siamrpn:  SiamRPN++ 完整模型封装
"""

from .backbone import build_backbone
from .cbam import CBAM
from .rpn import MultiRPN
from .neck import AdjustAllLayer
from .siamrpn import SiamRPNPP

__all__ = [
    "build_backbone",
    "CBAM",
    "MultiRPN",
    "AdjustAllLayer",
    "SiamRPNPP",
]
