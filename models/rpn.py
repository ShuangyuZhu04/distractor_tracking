"""
rpn.py — Region Proposal Network (区域候选网络)
=================================================
职责：基于模板特征与搜索特征的深度互相关 (Depth-wise Correlation)，
     输出分类得分图（前景/背景）和回归偏移图（边界框调整量）。

核心操作 — 深度互相关:
  将模板特征作为「卷积核」，在搜索特征上滑动计算相关性，
  得到响应图 (response map)，响应峰值即为目标可能位置。

设计说明：
  多尺度 RPN (MultiRPN) 对 layer2/3/4 三层特征分别计算互相关，
  最终加权融合得到最终的分类和回归结果。
"""

import logging
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class DepthwiseCorrelation(nn.Module):
    """
    深度互相关层。

    将模板特征 (kernel) 作为分组卷积的卷积核，
    在搜索特征 (search) 上执行分组卷积，实现逐通道互相关。

    输出 shape = (B, C, H_s - H_k + 1, W_s - W_k + 1)
    """

    def __init__(self):
        super().__init__()

    def forward(
        self, search: torch.Tensor, kernel: torch.Tensor
    ) -> torch.Tensor:
        """
        参数:
            search: 搜索区域特征, shape = (B, C, H_s, W_s)
            kernel: 模板特征,     shape = (B, C, H_k, W_k)

        返回:
            correlation: 互相关响应图, shape = (B, C, H_out, W_out)
        """
        batch_size = search.size(0)
        channels = search.size(1)

        # 分组卷积实现逐通道互相关
        # 将 batch 维度合并到 channel 维度，使用 groups=B*C
        search_reshaped = search.reshape(1, batch_size * channels, *search.shape[2:])
        kernel_reshaped = kernel.reshape(batch_size * channels, 1, *kernel.shape[2:])

        correlation = F.conv2d(
            search_reshaped, kernel_reshaped, groups=batch_size * channels
        )
        correlation = correlation.reshape(batch_size, channels, *correlation.shape[2:])

        return correlation


class SingleRPN(nn.Module):
    """
    单尺度 RPN 头。

    包含两个分支：
      - cls_branch: 分类分支（前景/背景二分类）
      - reg_branch: 回归分支（边界框偏移量 Δx, Δy, Δw, Δh）

    参数:
        in_channels (int):      输入通道数（经 Neck 调整后）
        cls_out_channels (int): 分类输出通道数（默认 2: 前景/背景）
        num_anchors (int):      每个位置的锚框数量
    """

    def __init__(
        self,
        in_channels: int = 256,
        cls_out_channels: int = 2,
        num_anchors: int = 5,
    ):
        super().__init__()

        self.correlation = DepthwiseCorrelation()

        # 分类分支: 互相关后接 1×1 卷积输出分类得分
        self.cls_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, cls_out_channels * num_anchors, kernel_size=1),
        )

        # 回归分支: 互相关后接 1×1 卷积输出偏移量
        self.reg_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, 4 * num_anchors, kernel_size=1),
        )

    def forward(
        self, template_feat: torch.Tensor, search_feat: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        参数:
            template_feat: 模板特征, shape = (B, C, H_t, W_t)
            search_feat:   搜索特征, shape = (B, C, H_s, W_s)

        返回:
            cls_score: 分类得分, shape = (B, 2*num_anchors, H_out, W_out)
            bbox_pred: 回归预测, shape = (B, 4*num_anchors, H_out, W_out)
        """
        # 深度互相关
        corr = self.correlation(search_feat, template_feat)

        # 分类 & 回归
        cls_score = self.cls_conv(corr)
        bbox_pred = self.reg_conv(corr)

        return cls_score, bbox_pred


class MultiRPN(nn.Module):
    """
    多尺度 RPN：对多层特征分别执行 SingleRPN，然后加权融合。

    参数:
        in_channels (int):      每层输入通道数（需 Neck 已统一）
        cls_out_channels (int): 分类输出通道数
        num_anchors (int):      每个位置的锚框数量
        num_layers (int):       参与融合的特征层数（默认 3: layer2/3/4）
    """

    def __init__(
        self,
        in_channels: int = 256,
        cls_out_channels: int = 2,
        num_anchors: int = 5,
        num_layers: int = 3,
    ):
        super().__init__()

        # 每层一个独立的 SingleRPN
        self.rpn_heads = nn.ModuleList([
            SingleRPN(in_channels, cls_out_channels, num_anchors)
            for _ in range(num_layers)
        ])

        # 各层融合权重（可学习参数）
        self.layer_weights = nn.Parameter(torch.ones(num_layers) / num_layers)

        logger.info(
            f"MultiRPN 初始化完成 | layers={num_layers}, "
            f"anchors={num_anchors}, in_ch={in_channels}"
        )

    def forward(
        self,
        template_feats: List[torch.Tensor],
        search_feats: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        参数:
            template_feats: 模板多尺度特征列表
            search_feats:   搜索多尺度特征列表

        返回:
            cls_score: 融合后的分类得分
            bbox_pred: 融合后的回归预测
        """
        # 归一化融合权重
        weights = F.softmax(self.layer_weights, dim=0)

        cls_scores = []
        bbox_preds = []

        for i, rpn_head in enumerate(self.rpn_heads):
            cls, reg = rpn_head(template_feats[i], search_feats[i])
            cls_scores.append(cls * weights[i])
            bbox_preds.append(reg * weights[i])

        #   使用空洞卷积 (replace_stride_with_dilation=[False,True,True]) 后,
        #   layer2/3/4 输出的空间分辨率已完全相同 (均为 stride=8),
        #   互相关结果天然就是 17×17, 无需插值对齐.
        cls_score = sum(cls_scores)
        bbox_pred = sum(bbox_preds)

        return cls_score, bbox_pred
