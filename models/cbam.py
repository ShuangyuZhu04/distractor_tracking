"""
cbam.py — CBAM 卷积块注意力模块
==================================
论文: Woo et al., "CBAM: Convolutional Block Attention Module", ECCV 2018

设计说明：
  CBAM 由「通道注意力」和「空间注意力」两个子模块串联而成：
    1) 通道注意力: 全局平均池化 + 全连接层 → 为每个通道分配权重
       作用: 强化目标关键特征通道，抑制背景噪声通道
    2) 空间注意力: 通道压缩 + 卷积 → 为每个空间位置分配权重
       作用: 定位目标核心区域，弱化无关背景区域

  在本项目中，CBAM 插入在搜索分支的 layer3 之后，目的是：
    - 为干扰物挖掘模块提供「关注度特征图」
    - 低关注度区域中的高相似度区域即为潜在干扰物
"""

import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class ChannelAttention(nn.Module):
    """
    通道注意力子模块。

    流程:
        输入特征 F ∈ (B, C, H, W)
        → 全局平均池化 → (B, C, 1, 1) ─┐
        → 全局最大池化 → (B, C, 1, 1) ─┤
        → 共享 MLP (FC → ReLU → FC)     │
        → 两路结果相加 → Sigmoid         │
        → 通道权重 Mc ∈ (B, C, 1, 1)   ←┘
        输出: F' = F ⊗ Mc  (逐通道加权)

    参数:
        in_channels (int):     输入特征通道数
        reduction_ratio (int): MLP 中间层压缩比（默认 16）
    """

    def __init__(self, in_channels: int, reduction_ratio: int = 16):
        super().__init__()

        mid_channels = max(in_channels // reduction_ratio, 8)  # 至少保留 8 通道

        # 共享 MLP: 两层全连接 (压缩 → 还原)
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, mid_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid_channels, in_channels, bias=False),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        参数:
            x: 输入特征, shape = (B, C, H, W)
        返回:
            通道注意力加权后的特征, shape = (B, C, H, W)
        """
        B, C, _, _ = x.shape

        # 全局平均池化 → (B, C)
        avg_pool = x.mean(dim=[2, 3])
        # 全局最大池化 → (B, C)
        max_pool = x.amax(dim=[2, 3])

        # 共享 MLP 计算通道权重
        avg_out = self.mlp(avg_pool)
        max_out = self.mlp(max_pool)

        # 相加 → Sigmoid → 通道权重
        channel_weight = self.sigmoid(avg_out + max_out)  # (B, C)
        channel_weight = channel_weight.unsqueeze(-1).unsqueeze(-1)  # (B, C, 1, 1)

        return x * channel_weight


class SpatialAttention(nn.Module):
    """
    空间注意力子模块。

    流程:
        输入特征 F' ∈ (B, C, H, W)
        → 通道维度平均池化 → (B, 1, H, W) ─┐
        → 通道维度最大池化 → (B, 1, H, W) ─┤
        → 拼接 → (B, 2, H, W)              │
        → 7×7 卷积 → Sigmoid               │
        → 空间权重 Ms ∈ (B, 1, H, W)      ←┘
        输出: F'' = F' ⊗ Ms  (逐位置加权)

    参数:
        kernel_size (int): 空间卷积核大小（默认 7，需为奇数）
    """

    def __init__(self, kernel_size: int = 7):
        super().__init__()
        assert kernel_size % 2 == 1, f"kernel_size 需为奇数，当前: {kernel_size}"

        padding = kernel_size // 2
        self.conv = nn.Conv2d(
            in_channels=2,
            out_channels=1,
            kernel_size=kernel_size,
            padding=padding,
            bias=False,
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        参数:
            x: 输入特征, shape = (B, C, H, W)
        返回:
            空间注意力加权后的特征, shape = (B, C, H, W)
            以及空间注意力图 (用于干扰物挖掘), shape = (B, 1, H, W)
        """
        # 沿通道维度压缩
        avg_pool = x.mean(dim=1, keepdim=True)    # (B, 1, H, W)
        max_pool = x.amax(dim=1, keepdim=True)    # (B, 1, H, W)

        # 拼接 → 卷积 → Sigmoid
        combined = torch.cat([avg_pool, max_pool], dim=1)  # (B, 2, H, W)
        spatial_weight = self.sigmoid(self.conv(combined))  # (B, 1, H, W)

        return x * spatial_weight, spatial_weight


class CBAM(nn.Module):
    """
    CBAM 完整模块 = 通道注意力 → 空间注意力（串联）。

    额外返回空间注意力图 (attention_map)，供干扰物挖掘模块使用。

    参数:
        in_channels (int):       输入特征通道数
        reduction_ratio (int):   通道注意力压缩比
        spatial_kernel_size (int): 空间注意力卷积核大小
    """

    def __init__(
        self,
        in_channels: int,
        reduction_ratio: int = 16,
        spatial_kernel_size: int = 7,
    ):
        super().__init__()
        self.channel_attn = ChannelAttention(in_channels, reduction_ratio)
        self.spatial_attn = SpatialAttention(spatial_kernel_size)

        logger.info(
            f"CBAM 初始化完成 | channels={in_channels}, "
            f"reduction={reduction_ratio}, spatial_k={spatial_kernel_size}"
        )

    def forward(self, x: torch.Tensor):
        """
        参数:
            x: 输入特征, shape = (B, C, H, W)

        返回:
            refined_features: CBAM 加权后的特征, shape = (B, C, H, W)
            attention_map:    空间注意力图, shape = (B, 1, H, W)
                              值域 [0, 1]，高值 = 高关注区域（可能是目标）
                                          低值 = 低关注区域（背景/潜在干扰物）
        """
        # 第一步: 通道注意力
        x = self.channel_attn(x)

        # 第二步: 空间注意力（同时输出注意力图）
        refined_features, attention_map = self.spatial_attn(x)

        return refined_features, attention_map
