"""
features.py — HOG 特征提取工具
=================================
封装 OpenCV 的 HOG 描述子，提供统一的特征提取接口。
在干扰物挖掘模块中用于计算候选区域与目标的形状相似度。
"""

import logging
from typing import Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class HOGExtractor:
    """
    HOG 特征提取器。

    参数:
        win_size (tuple):      检测窗口大小
        block_size (tuple):    块大小
        block_stride (tuple):  块步长
        cell_size (tuple):     单元格大小
        nbins (int):           方向梯度直方图的 bin 数
    """

    def __init__(
        self,
        win_size: Tuple[int, int] = (64, 128),
        block_size: Tuple[int, int] = (16, 16),
        block_stride: Tuple[int, int] = (8, 8),
        cell_size: Tuple[int, int] = (8, 8),
        nbins: int = 9,
    ):
        self.win_size = win_size
        self.hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, nbins)

        # 计算输出特征维度
        blocks_per_window = (
            (win_size[0] - block_size[0]) // block_stride[0] + 1,
            (win_size[1] - block_size[1]) // block_stride[1] + 1,
        )
        cells_per_block = (
            block_size[0] // cell_size[0],
            block_size[1] // cell_size[1],
        )
        self.feature_dim = (
            blocks_per_window[0] * blocks_per_window[1]
            * cells_per_block[0] * cells_per_block[1]
            * nbins
        )

        logger.debug(f"HOGExtractor 初始化 | win_size={win_size}, dim={self.feature_dim}")

    def extract(self, image: np.ndarray) -> np.ndarray:
        """
        提取图像的 HOG 特征。

        参数:
            image: 输入图像 (H, W, 3) BGR 或 (H, W) 灰度

        返回:
            feature: HOG 特征向量, shape = (feature_dim,)
        """
        # 缩放到窗口大小
        resized = cv2.resize(image, self.win_size)
        if len(resized.shape) == 3:
            resized = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

        descriptor = self.hog.compute(resized)
        if descriptor is None:
            return np.zeros(self.feature_dim)

        return descriptor.flatten()

    def batch_extract(self, patches: list) -> np.ndarray:
        """
        批量提取 HOG 特征。

        参数:
            patches: 图像区域列表

        返回:
            features: shape = (N, feature_dim)
        """
        return np.stack([self.extract(p) for p in patches])

    @staticmethod
    def cosine_similarity(feat_a: np.ndarray, feat_b: np.ndarray) -> float:
        """计算两个特征向量的余弦相似度。"""
        norm_a = np.linalg.norm(feat_a)
        norm_b = np.linalg.norm(feat_b)
        if norm_a < 1e-8 or norm_b < 1e-8:
            return 0.0
        return float(np.clip(np.dot(feat_a, feat_b) / (norm_a * norm_b), 0.0, 1.0))
