"""
neck.py — 特征调整层 (Adjust Layer)
=====================================
职责：将骨干网络输出的多尺度特征统一调整到相同的通道数，
     以便后续 RPN 头进行深度互相关计算。

在 SiamRPN++ 中，模板分支和搜索分支共享骨干网络，
但各层输出通道数不同（512/1024/2048），
需要通过 1×1 卷积统一为 rpn.in_channels（默认 256）。
"""

import logging
from typing import List

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class AdjustLayer(nn.Module):
    """
    单层特征调整：1×1 卷积 + BN。

    参数:
        in_channels (int):  输入通道数（来自骨干网络某一层）
        out_channels (int): 输出通道数（统一为 RPN 输入通道数）
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.adjust = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.adjust(x)


class AdjustAllLayer(nn.Module):
    """
    多层特征调整：对骨干网络输出的每一层特征分别做通道调整。

    参数:
        in_channels_list (list[int]):  各层输入通道数，如 [512, 1024, 2048]
        out_channels (int):            统一输出通道数，如 256
    """

    def __init__(self, in_channels_list: List[int], out_channels: int):
        super().__init__()
        self.adjusts = nn.ModuleList([
            AdjustLayer(in_ch, out_channels) for in_ch in in_channels_list
        ])
        logger.info(
            f"AdjustAllLayer 初始化完成 | "
            f"输入通道数: {in_channels_list} → 统一输出: {out_channels}"
        )

    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        参数:
            features: 多尺度特征列表, 每个元素 shape = (B, C_i, H_i, W_i)

        返回:
            adjusted: 调整后的特征列表, 每个元素 shape = (B, out_channels, H_i, W_i)
        """
        return [adjust(feat) for adjust, feat in zip(self.adjusts, features)]
