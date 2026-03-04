"""
core 包 — 核心算法逻辑
========================
包含跟踪主控器、干扰物管理器和融合优化模块。

调用关系:
  Tracker (跟踪主控)
    ├── SiamRPN++  (模型推理)
    ├── DistractorManager (干扰物挖掘与管理)
    └── FusionModule (干扰物信息融合优化)
"""

from .tracker import Tracker
from .distractor import DistractorManager, DistractorInfo, RiskLevel
from .fusion import FusionModule

__all__ = [
    "Tracker",
    "DistractorManager",
    "DistractorInfo",
    "RiskLevel",
    "FusionModule",
]
