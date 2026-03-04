"""
transforms.py — 数据增强与预处理
===================================
训练时使用的数据增强策略，包括：
  - 随机平移、缩放
  - 颜色抖动
  - 随机水平翻转
  - 模板/搜索区域对的裁剪与缩放
"""

import logging
from typing import Dict, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class TrackingTransform:
    """
    跟踪任务数据增强。

    在训练时对模板图像和搜索图像进行一致的几何变换与颜色增强，
    以提升模型的泛化能力。

    参数:
        template_size (int): 模板裁剪尺寸（默认 127）
        search_size (int):   搜索区域裁剪尺寸（默认 255）
        context_amount (float): 上下文扩展比例
        color_jitter (float):   颜色抖动幅度 [0, 1]
        flip_prob (float):      水平翻转概率
        scale_range (tuple):    随机缩放范围
        shift_range (int):      随机平移范围（像素）
    """

    def __init__(
        self,
        template_size: int = 127,
        search_size: int = 255,
        context_amount: float = 0.5,
        color_jitter: float = 0.1,
        flip_prob: float = 0.5,
        scale_range: Tuple[float, float] = (0.9, 1.1),
        shift_range: int = 8,
    ):
        self.template_size = template_size
        self.search_size = search_size
        self.context_amount = context_amount
        self.color_jitter = color_jitter
        self.flip_prob = flip_prob
        self.scale_range = scale_range
        self.shift_range = shift_range

    def __call__(
        self,
        template_img: np.ndarray,
        search_img: np.ndarray,
        template_bbox: np.ndarray,
        search_bbox: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """
        对模板-搜索图像对执行数据增强。

        参数:
            template_img:  模板帧图像 (H, W, 3)
            search_img:    搜索帧图像 (H, W, 3)
            template_bbox: 模板目标框 [x, y, w, h]
            search_bbox:   搜索目标框 [x, y, w, h]

        返回:
            dict:
                "template":      增强后的模板图像 (template_size, template_size, 3)
                "search":        增强后的搜索图像 (search_size, search_size, 3)
                "template_bbox": 变换后的模板标注框
                "search_bbox":   变换后的搜索标注框
        """
        # 裁剪模板区域
        template_patch = self._crop_and_resize(
            template_img, template_bbox, self.template_size
        )

        # 裁剪搜索区域（带随机平移和缩放增强）
        augmented_bbox = self._augment_bbox(search_bbox)
        search_patch = self._crop_and_resize(
            search_img, augmented_bbox, self.search_size
        )

        # 颜色增强
        if self.color_jitter > 0:
            template_patch = self._apply_color_jitter(template_patch)
            search_patch = self._apply_color_jitter(search_patch)

        # 随机水平翻转
        if np.random.random() < self.flip_prob:
            template_patch = cv2.flip(template_patch, 1)
            search_patch = cv2.flip(search_patch, 1)

        return {
            "template": template_patch,
            "search": search_patch,
            "template_bbox": template_bbox,
            "search_bbox": search_bbox,
        }

    def _crop_and_resize(
        self,
        img: np.ndarray,
        bbox: np.ndarray,
        output_size: int,
    ) -> np.ndarray:
        """以目标框为中心裁剪并缩放到指定尺寸。"""
        x, y, w, h = bbox
        cx, cy = x + w / 2, y + h / 2

        context = self.context_amount * (w + h)
        base_crop = np.sqrt((w + context) * (h + context))

        # 动态引入尺度因子 (output_size / template_size)
        # 如果 output_size 是 255，那么 crop_size 就会翻倍，保证目标缩放后占据相同的相对像素
        crop_size = int(base_crop * (output_size / self.template_size))
        #crop_size = max(1, crop_size)  # 防止边界报错为0

        x1 = int(cx - crop_size / 2)
        y1 = int(cy - crop_size / 2)
        x2 = x1 + crop_size
        y2 = y1 + crop_size

        H, W = img.shape[:2]
        pad_left = max(0, -x1)
        pad_top = max(0, -y1)
        pad_right = max(0, x2 - W)
        pad_bottom = max(0, y2 - H)

        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(W, x2), min(H, y2)

        patch = img[y1:y2, x1:x2]
        if any([pad_left, pad_top, pad_right, pad_bottom]):
            patch = cv2.copyMakeBorder(
                patch, pad_top, pad_bottom, pad_left, pad_right,
                cv2.BORDER_CONSTANT, value=(0, 0, 0)
            )

        return cv2.resize(patch, (output_size, output_size))

    def _augment_bbox(self, bbox: np.ndarray) -> np.ndarray:
        """对搜索框施加随机平移和缩放。"""
        x, y, w, h = bbox.copy()

        # 随机平移
        dx = np.random.randint(-self.shift_range, self.shift_range + 1)
        dy = np.random.randint(-self.shift_range, self.shift_range + 1)

        # 随机缩放
        scale = np.random.uniform(*self.scale_range)

        new_w = w * scale
        new_h = h * scale
        new_x = x + dx + (w - new_w) / 2
        new_y = y + dy + (h - new_h) / 2

        return np.array([new_x, new_y, new_w, new_h])

    def _apply_color_jitter(self, img: np.ndarray) -> np.ndarray:
        """随机颜色抖动（亮度、对比度、饱和度）。"""
        img = img.astype(np.float32)

        # 亮度
        brightness = 1.0 + np.random.uniform(-self.color_jitter, self.color_jitter)
        img *= brightness

        # 对比度
        contrast = 1.0 + np.random.uniform(-self.color_jitter, self.color_jitter)
        mean = img.mean()
        img = (img - mean) * contrast + mean

        return np.clip(img, 0, 255).astype(np.uint8)
