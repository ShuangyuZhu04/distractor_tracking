"""
anchor.py — Anchor 目标生成器 (AnchorTargetGenerator)
=======================================================
负责 SiamRPN++ 训练时的标签分配：
  1. 在 score map 上生成 anchor 网格
  2. 计算每个 anchor 与 GT bbox 的 IoU
  3. 基于 IoU 阈值分配正样本 / 负样本 / 忽略样本
  4. 为正样本计算回归目标 (dx, dy, dw, dh)
  5. 执行正负样本采样平衡

参考: SiamRPN++ (Li et al., CVPR 2019) 的 anchor 匹配策略。

核心公式:
  anchor 中心坐标:
    x_anchor = origin + j * stride,   j = 0, 1, ..., score_size - 1
    y_anchor = origin + i * stride,   i = 0, 1, ..., score_size - 1
    origin = (search_size - 1) / 2 - (score_size - 1) / 2 * stride

  回归目标:
    dx = (gt_cx - anchor_cx) / anchor_w
    dy = (gt_cy - anchor_cy) / anchor_h
    dw = log(gt_w / anchor_w)
    dh = log(gt_h / anchor_h)
"""

import logging
from typing import Dict

import numpy as np

logger = logging.getLogger(__name__)


class AnchorTargetGenerator:
    """
    Anchor 目标生成器。

    在 SiamRPN++ 中，score map 的每个空间位置对应 A 个 anchor，
    每个 anchor 由 (cx, cy, w, h) 定义。训练时需要为每个 anchor
    分配标签（正/负/忽略）和回归目标。

    参数:
        score_size (int):         score map 空间尺寸（如 17）
        stride (int):             anchor 步长（如 8）
        search_size (int):        搜索图像尺寸（如 255）
        anchor_scales (list):     anchor 尺度列表（如 [8]）
        anchor_ratios (list):     anchor 宽高比列表（如 [0.33, 0.5, 1.0, 2.0, 3.0]）
        pos_iou_thr (float):      正样本 IoU 阈值
        neg_iou_thr (float):      负样本 IoU 阈值
        total_sample_num (int):   每帧采样的 anchor 总数
        pos_ratio (float):        正样本在采样中的占比上限
    """

    def __init__(
        self,
        score_size: int = 17,
        stride: int = 8,
        search_size: int = 255,
        anchor_scales: list = None,
        anchor_ratios: list = None,
        pos_iou_thr: float = 0.6,
        neg_iou_thr: float = 0.3,
        total_sample_num: int = 64,
        pos_ratio: float = 0.25,
    ):
        self.score_size = score_size
        self.stride = stride
        self.search_size = search_size
        self.anchor_scales = anchor_scales or [8]
        self.anchor_ratios = anchor_ratios or [0.33, 0.5, 1.0, 2.0, 3.0]
        self.pos_iou_thr = pos_iou_thr
        self.neg_iou_thr = neg_iou_thr
        self.total_sample_num = total_sample_num
        self.pos_ratio = pos_ratio

        # 每个位置的 anchor 数量: A = len(scales) * len(ratios)
        self.num_anchors = len(self.anchor_scales) * len(self.anchor_ratios)

        # anchor 总数 = A * S * S
        self.total_anchors = self.num_anchors * score_size * score_size

        # 预生成 anchor 网格（只算一次，整个训练过程复用）
        self.anchors = self._generate_anchors()  # (total_anchors, 4) [cx,cy,w,h]

        logger.info(
            f"AnchorTargetGenerator 初始化 | "
            f"score_size={score_size}, stride={stride}, "
            f"num_anchors_per_pos={self.num_anchors}, "
            f"total_anchors={self.total_anchors}"
        )

    # ==========================================================
    #  Anchor 网格生成
    # ==========================================================

    def _generate_base_anchors(self) -> np.ndarray:
        """
        生成一个位置上的 A 个 base anchor（以原点 (0,0) 为中心）。

        面积 = (stride * scale)^2，按宽高比分配:
          w = base_size * sqrt(ratio)
          h = base_size / sqrt(ratio)

        返回:
            base_anchors: shape = (A, 4), 每行 [cx=0, cy=0, w, h]
        """
        anchors = []
        for scale in self.anchor_scales:
            base_size = self.stride * scale  # 如 8 × 8 = 64 像素
            for ratio in self.anchor_ratios:
                w = base_size * np.sqrt(ratio)
                h = base_size / np.sqrt(ratio)
                anchors.append([0.0, 0.0, w, h])

        return np.array(anchors, dtype=np.float64)

    def _generate_anchors(self) -> np.ndarray:
        """
        在 score map 上铺设完整的 anchor 网格。

        score map 位置 (i, j) → 搜索图像坐标:
          x = origin + j * stride
          y = origin + i * stride
          origin = (search_size-1)/2 - (score_size-1)/2 * stride
          例: (255-1)/2 - (17-1)/2 * 8 = 127 - 64 = 63

        返回:
            anchors: shape = (A * S * S, 4), [cx, cy, w, h]

        内存排列顺序 (与 RPN 输出对齐):
          anchor_type_0 @ (0,0), anchor_type_0 @ (0,1), ...,
          anchor_type_0 @ (S-1,S-1),
          anchor_type_1 @ (0,0), ...,
          anchor_type_{A-1} @ (S-1,S-1)
        """
        base_anchors = self._generate_base_anchors()

        origin = (
            (self.search_size - 1) / 2.0
            - (self.score_size - 1) / 2.0 * self.stride
        )

        shift_x = origin + np.arange(self.score_size) * self.stride
        shift_y = origin + np.arange(self.score_size) * self.stride
        mesh_x, mesh_y = np.meshgrid(shift_x, shift_y)

        mesh_x = mesh_x.flatten()  # (S*S,)
        mesh_y = mesh_y.flatten()

        num_pos = self.score_size * self.score_size
        all_anchors = np.zeros((self.num_anchors, num_pos, 4), dtype=np.float64)

        for a in range(self.num_anchors):
            all_anchors[a, :, 0] = mesh_x      # cx
            all_anchors[a, :, 1] = mesh_y      # cy
            all_anchors[a, :, 2] = base_anchors[a, 2]  # w (broadcast)
            all_anchors[a, :, 3] = base_anchors[a, 3]  # h

        # (A, S*S, 4) → (A*S*S, 4)
        return all_anchors.reshape(-1, 4)

    # ==========================================================
    #  IoU 计算
    # ==========================================================

    @staticmethod
    def _compute_iou(anchors: np.ndarray, gt_bbox: np.ndarray) -> np.ndarray:
        """
        计算所有 anchor 与单个 GT bbox 的 IoU。

        参数:
            anchors: (N, 4) [cx, cy, w, h]
            gt_bbox: (4,)   [cx, cy, w, h]

        返回:
            ious: (N,)
        """
        # → [x1, y1, x2, y2]
        a_x1 = anchors[:, 0] - anchors[:, 2] / 2
        a_y1 = anchors[:, 1] - anchors[:, 3] / 2
        a_x2 = anchors[:, 0] + anchors[:, 2] / 2
        a_y2 = anchors[:, 1] + anchors[:, 3] / 2

        g_x1 = gt_bbox[0] - gt_bbox[2] / 2
        g_y1 = gt_bbox[1] - gt_bbox[3] / 2
        g_x2 = gt_bbox[0] + gt_bbox[2] / 2
        g_y2 = gt_bbox[1] + gt_bbox[3] / 2

        inter_w = np.maximum(0, np.minimum(a_x2, g_x2) - np.maximum(a_x1, g_x1))
        inter_h = np.maximum(0, np.minimum(a_y2, g_y2) - np.maximum(a_y1, g_y1))
        inter_area = inter_w * inter_h

        union_area = anchors[:, 2] * anchors[:, 3] + gt_bbox[2] * gt_bbox[3] - inter_area

        return inter_area / np.maximum(union_area, 1e-8)

    # ==========================================================
    #  回归目标编码
    # ==========================================================

    @staticmethod
    def _encode_regression(
        anchors: np.ndarray, gt_bbox: np.ndarray
    ) -> np.ndarray:
        """
        编码回归目标 (dx, dy, dw, dh)。

        参数:
            anchors: (N, 4) [cx, cy, w, h]
            gt_bbox: (4,)   [cx, cy, w, h]

        返回:
            targets: (N, 4) [dx, dy, dw, dh]
        """
        dx = (gt_bbox[0] - anchors[:, 0]) / np.maximum(anchors[:, 2], 1e-8)
        dy = (gt_bbox[1] - anchors[:, 1]) / np.maximum(anchors[:, 3], 1e-8)
        dw = np.log(np.maximum(gt_bbox[2], 1e-8) / np.maximum(anchors[:, 2], 1e-8))
        dh = np.log(np.maximum(gt_bbox[3], 1e-8) / np.maximum(anchors[:, 3], 1e-8))

        return np.stack([dx, dy, dw, dh], axis=1)

    # ==========================================================
    #  正负样本采样
    # ==========================================================

    def _sample_anchors(self, labels: np.ndarray) -> np.ndarray:
        """
        正负样本平衡采样。

        策略:
          pos_num = min(实际正样本, total_sample_num * pos_ratio)
          neg_num = total_sample_num - pos_num
          多余的正/负样本设为忽略 (-1)
        """
        labels = labels.copy()

        pos_idx = np.where(labels == 1)[0]
        max_pos = int(self.total_sample_num * self.pos_ratio)

        if len(pos_idx) > max_pos:
            drop = np.random.choice(pos_idx, size=len(pos_idx) - max_pos, replace=False)
            labels[drop] = -1
            pos_idx = np.where(labels == 1)[0]

        neg_idx = np.where(labels == 0)[0]
        max_neg = self.total_sample_num - len(pos_idx)

        if len(neg_idx) > max_neg:
            drop = np.random.choice(neg_idx, size=len(neg_idx) - max_neg, replace=False)
            labels[drop] = -1

        return labels

    # ==========================================================
    #  主入口
    # ==========================================================

    def generate(
        self, gt_bbox_in_search: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        根据 GT bbox 为所有 anchor 生成分类标签和回归目标。

        参数:
            gt_bbox_in_search: GT 在 255×255 搜索图像中的坐标 [cx, cy, w, h]

        返回:
            dict:
                "cls_label":  (total_anchors,)   {-1, 0, 1}
                "reg_label":  (total_anchors, 4) [dx, dy, dw, dh]
                "pos_mask":   (total_anchors,)   bool
                "neg_mask":   (total_anchors,)   bool
        """
        gt = gt_bbox_in_search.astype(np.float64)

        # Step 1: IoU
        ious = self._compute_iou(self.anchors, gt)

        # Step 2: 分配标签
        labels = np.full(self.total_anchors, -1, dtype=np.int64)
        labels[ious >= self.pos_iou_thr] = 1
        labels[ious < self.neg_iou_thr] = 0

        # 保证至少一个正样本
        if np.sum(labels == 1) == 0:
            labels[np.argmax(ious)] = 1

        # Step 3: 采样
        labels = self._sample_anchors(labels)

        # Step 4: 回归目标（仅正样本有效）
        reg_targets = np.zeros((self.total_anchors, 4), dtype=np.float64)
        pos_mask = labels == 1
        if np.any(pos_mask):
            reg_targets[pos_mask] = self._encode_regression(self.anchors[pos_mask], gt)

        return {
            "cls_label": labels,
            "reg_label": reg_targets,
            "pos_mask": pos_mask,
            "neg_mask": labels == 0,
        }