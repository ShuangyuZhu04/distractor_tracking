"""
fusion.py — 融合优化模块 (FusionModule)
==========================================
对应开题报告的：    「干扰物信息与跟踪算法的有效融合」

核心设计 — "动态反馈 + 策略适配":
  根据干扰物的风险等级，采取差异化的跟踪修正策略：

  ┌────────────────┬──────────────────────────────────────┐
  │   风险等级     │          应对策略                     │
  ├────────────────┼──────────────────────────────────────┤
  │ HIGH_SIMILAR   │ 缩小搜索范围，避免误匹配             │
  │ OCCLUSION      │ 基于干扰物位置修正跟踪框坐标         │
  │ DYNAMIC        │ 预测性偏移，预防被干扰物"带偏"       │
  └────────────────┴──────────────────────────────────────┘

另外提供干扰物权重损失函数，用于训练时增强模型对干扰物的辨别能力。
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from .distractor import DistractorInfo, RiskLevel

logger = logging.getLogger(__name__)


@dataclass
class TrackingAdjustment:
    """
    融合模块输出的跟踪调整指令。

    属性:
        search_region_scale (float): 搜索区域缩放因子（<1 表示收缩）
        bbox_offset (np.ndarray):    跟踪框偏移量 [dx, dy]（像素）
        confidence_penalty (float):  置信度惩罚因子 [0, 1]
        trigger_redetect (bool):     是否触发目标重检测
        strategy_name (str):         所采用的策略名称（用于日志和可视化）
    """
    search_region_scale: float = 1.0
    bbox_offset: np.ndarray = None
    confidence_penalty: float = 0.0
    trigger_redetect: bool = False
    strategy_name: str = "none"

    def __post_init__(self):
        if self.bbox_offset is None:
            self.bbox_offset = np.array([0.0, 0.0])


class FusionModule:
    """
    融合优化模块 — 将干扰物信息转化为跟踪策略调整。

    参数:
        cfg (dict): 配置字典（fusion 节）
    """

    def __init__(self, cfg: dict):
        fusion_cfg = cfg.get("fusion", {})

        self.shrink_ratio: float = fusion_cfg.get("shrink_ratio", 0.7)
        self.redetect_score_thr: float = fusion_cfg.get("redetect_score_thr", 0.5)
        self.redetect_expand: float = fusion_cfg.get("redetect_expand", 2.0)
        self.distractor_loss_weight: float = fusion_cfg.get("distractor_loss_weight", 1.5)

        logger.info(
            f"FusionModule 初始化完成 | "
            f"shrink_ratio={self.shrink_ratio}, "
            f"redetect_thr={self.redetect_score_thr}"
        )

    # ==========================================================
    #  核心接口：生成跟踪调整指令
    # ==========================================================

    def compute_adjustment(
        self,
        target_bbox: np.ndarray,
        target_score: float,
        distractors: List[DistractorInfo],
    ) -> TrackingAdjustment:
        """
        根据干扰物信息和当前跟踪状态，计算跟踪调整指令。

        优先级规则（高风险策略优先）：
          1. 若存在 HIGH_SIMILAR 干扰物 → 执行搜索范围收缩
          2. 若存在 OCCLUSION 干扰物   → 执行跟踪框位置修正
          3. 若存在 DYNAMIC 干扰物     → 执行预测性偏移
          4. 若置信度过低 + 存在高风险干扰物 → 触发目标重检测

        参数:
            target_bbox (np.ndarray): 当前目标框 [x1, y1, x2, y2]
            target_score (float):     当前跟踪置信度 [0, 1]
            distractors (list):       当前帧干扰物列表

        返回:
            TrackingAdjustment: 跟踪调整指令
        """
        if not distractors:
            return TrackingAdjustment(strategy_name="no_distractor")

        adjustment = TrackingAdjustment()

        # --- 按风险等级分组 ---
        high_similar = [d for d in distractors if d.risk_level == RiskLevel.HIGH_SIMILAR]
        occlusion = [d for d in distractors if d.risk_level == RiskLevel.OCCLUSION]
        dynamic = [d for d in distractors if d.risk_level == RiskLevel.DYNAMIC]

        strategies_applied = []

        # 策略 1: 高相似干扰物 → 搜索范围收缩
        if high_similar:
            adjustment.search_region_scale = self._compute_shrink_scale(
                target_bbox, high_similar
            )
            strategies_applied.append("shrink_search")

        # 策略 2: 遮挡型干扰物 → 跟踪框位置修正
        if occlusion:
            offset = self._compute_occlusion_offset(target_bbox, occlusion)
            adjustment.bbox_offset = offset
            strategies_applied.append("occlusion_offset")

        # 策略 3: 动态干扰物 → 预测性偏移
        if dynamic:
            preventive_offset = self._compute_dynamic_offset(target_bbox, dynamic)
            adjustment.bbox_offset = adjustment.bbox_offset + preventive_offset
            strategies_applied.append("dynamic_offset")

        # 策略 4: 低置信度 + 高风险干扰物 → 触发重检测
        has_high_risk = len(high_similar) > 0 or len(occlusion) > 0
        if target_score < self.redetect_score_thr and has_high_risk:
            adjustment.trigger_redetect = True
            adjustment.search_region_scale = self.redetect_expand
            strategies_applied.append("redetect")

        adjustment.strategy_name = "+".join(strategies_applied) or "none"

        logger.debug(
            f"融合调整: score={target_score:.3f}, "
            f"策略=[{adjustment.strategy_name}], "
            f"scale={adjustment.search_region_scale:.2f}, "
            f"offset={adjustment.bbox_offset}"
        )

        return adjustment

    # ==========================================================
    #  策略实现
    # ==========================================================

    def _compute_shrink_scale(
        self,
        target_bbox: np.ndarray,
        high_similar_distractors: List[DistractorInfo],
    ) -> float:
        """
        计算搜索区域收缩比例。

        若干扰物为"高相似干扰物"，缩小跟踪框搜索范围以避免误匹配。

        逻辑:
          收缩比例与最近高相似干扰物的距离成正比（越近越需要收缩）。

        返回:
            scale: 搜索区域缩放因子 (0, 1]
        """
        tcx = (target_bbox[0] + target_bbox[2]) / 2
        tcy = (target_bbox[1] + target_bbox[3]) / 2

        min_dist = float("inf")
        for d in high_similar_distractors:
            dcx = (d.bbox[0] + d.bbox[2]) / 2
            dcy = (d.bbox[1] + d.bbox[3]) / 2
            dist = np.sqrt((tcx - dcx) ** 2 + (tcy - dcy) ** 2)
            min_dist = min(min_dist, dist)

        # 距离越近，收缩越多；距离 > 200 像素时不收缩
        if min_dist > 200:
            return 1.0

        scale = self.shrink_ratio + (1.0 - self.shrink_ratio) * (min_dist / 200.0)
        return float(np.clip(scale, self.shrink_ratio, 1.0))

    def _compute_occlusion_offset(
        self,
        target_bbox: np.ndarray,
        occlusion_distractors: List[DistractorInfo],
    ) -> np.ndarray:
        """
        计算遮挡场景下的跟踪框位置修正偏移量。

        若干扰物为"遮挡型干扰物"，基于干扰物位置修正跟踪框坐标，避免跟踪漂移。

        逻辑:
          将跟踪框向远离遮挡干扰物的方向微调（偏移量与遮挡程度成正比）。

        返回:
            offset: [dx, dy] 偏移量（像素）
        """
        tcx = (target_bbox[0] + target_bbox[2]) / 2
        tcy = (target_bbox[1] + target_bbox[3]) / 2

        total_dx, total_dy = 0.0, 0.0
        for d in occlusion_distractors:
            dcx = (d.bbox[0] + d.bbox[2]) / 2
            dcy = (d.bbox[1] + d.bbox[3]) / 2

            # 从干扰物指向目标的方向
            dx = tcx - dcx
            dy = tcy - dcy
            norm = np.sqrt(dx ** 2 + dy ** 2) + 1e-6

            # 偏移量与 IoU 成正比（遮挡越严重偏移越多）
            magnitude = d.iou_with_target * 10.0  # 最大约 10 像素
            total_dx += (dx / norm) * magnitude
            total_dy += (dy / norm) * magnitude

        return np.array([total_dx, total_dy])

    def _compute_dynamic_offset(
        self,
        target_bbox: np.ndarray,
        dynamic_distractors: List[DistractorInfo],
    ) -> np.ndarray:
        """
        计算动态干扰物场景下的预测性偏移。

        若干扰物向跟踪框方向移动，提前将跟踪框向远离干扰物的方向偏移，预防跟踪框被干扰物"带偏"。

        返回:
            offset: [dx, dy] 预测性偏移量（像素）
        """
        tcx = (target_bbox[0] + target_bbox[2]) / 2
        tcy = (target_bbox[1] + target_bbox[3]) / 2

        total_dx, total_dy = 0.0, 0.0
        for d in dynamic_distractors:
            dcx = (d.bbox[0] + d.bbox[2]) / 2
            dcy = (d.bbox[1] + d.bbox[3]) / 2

            # 从干扰物指向目标的方向（干扰物的运动趋势方向）
            dx = tcx - dcx
            dy = tcy - dcy
            norm = np.sqrt(dx ** 2 + dy ** 2) + 1e-6

            # 偏移量与干扰物运动速度成正比
            speed_factor = min(d.displacement / 50.0, 1.0)  # 归一化
            magnitude = speed_factor * 5.0  # 最大约 5 像素预防性偏移
            total_dx += (dx / norm) * magnitude
            total_dy += (dy / norm) * magnitude

        return np.array([total_dx, total_dy])

    # ==========================================================
    #  损失函数（训练时使用）
    # ==========================================================

    def compute_distractor_loss(
        self,
        cls_score: torch.Tensor,
        distractor_masks: torch.Tensor,
        target_labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        干扰物加权损失函数。

        在模型训练时，对干扰物区域的预测误差赋予更高权重，增强模型对干扰物的辨别能力。

        参数:
            cls_score (Tensor):       分类预测得分, shape = (B, 2, H, W)
            distractor_masks (Tensor): 干扰物区域掩码, shape = (B, H, W)
                                       值 = 1 表示干扰物区域
            target_labels (Tensor):   目标标签, shape = (B, H, W)
                                       值 = 1 表示正样本, 0 表示负样本

        返回:
            loss: 加权交叉熵损失
        """
        # 构建权重图: 干扰物区域权重更高
        weight_map = torch.ones_like(target_labels, dtype=torch.float32)
        weight_map[distractor_masks > 0] = self.distractor_loss_weight

        # 交叉熵损失
        loss = nn.functional.cross_entropy(
            cls_score, target_labels.long(), reduction="none"
        )

        # 加权求均值
        weighted_loss = (loss * weight_map).mean()

        return weighted_loss
