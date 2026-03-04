"""
distractor.py — 干扰物管理器 (DistractorManager)
====================================================
本文件是核心创新模块，对应问题：    「背景干扰物的精准定位与分类」

核心思路 — "主动挖掘 + 风险分级":
  1. 利用 CBAM 输出的空间注意力图定位「低关注度背景区域」
  2. 在背景区域中，通过 HOG + CNN 多特征相似度计算，
     筛选出与目标相似度高的区域作为「潜在干扰物」
  3. 按「相似度 + 运动特性 + 遮挡程度」将干扰物划分为三个风险等级:
     - HIGH_SIMILAR:  高相似干扰物（特征相似度 ≥ 阈值）
     - DYNAMIC:       动态干扰物（帧间位移 ≥ 阈值）
     - OCCLUSION:     遮挡型干扰物（与目标 IoU ≥ 阈值）

干扰物信息流:
  CBAM attention_map → 背景区域提取 → 候选区域生成
  → HOG 相似度 + CNN 相似度 → 加权融合 → 风险分级 → DistractorInfo 列表
"""

import logging
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.ops  # roi_align 用于从特征图中精确提取候选区域特征

logger = logging.getLogger(__name__)


# ============================================================
#  数据结构定义
# ============================================================

class RiskLevel(IntEnum):
    """
    干扰物风险等级枚举（对应开题报告的三级分类）。
    数值越大风险越高，用于后续融合模块确定应对策略优先级。
    """
    LOW = 0             # 低风险（普通背景区域，无需特殊处理）
    DYNAMIC = 1         # 动态干扰物（帧间位移 ≥ 阈值）
    OCCLUSION = 2       # 遮挡型干扰物（与目标重叠度 ≥ 阈值）
    HIGH_SIMILAR = 3    # 高相似干扰物（特征相似度 ≥ 阈值，最高风险）


@dataclass
class DistractorInfo:
    """
    单个干扰物的完整信息描述。

    属性:
        bbox (np.ndarray):       干扰物边界框 [x1, y1, x2, y2]（像素坐标）
        risk_level (RiskLevel):  风险等级
        similarity (float):     与目标的综合特征相似度 [0, 1]
        hog_similarity (float): HOG 特征余弦相似度
        cnn_similarity (float): CNN 特征欧氏距离归一化相似度
        displacement (float):   帧间位移（像素），用于判断是否为动态干扰物
        iou_with_target (float): 与目标框的 IoU，用于判断遮挡
        frame_id (int):          检测到该干扰物的帧号
        track_id (int):          干扰物跨帧跟踪 ID（-1 表示新出现）
    """
    bbox: np.ndarray
    risk_level: RiskLevel = RiskLevel.LOW
    similarity: float = 0.0
    hog_similarity: float = 0.0
    cnn_similarity: float = 0.0
    displacement: float = 0.0
    iou_with_target: float = 0.0
    frame_id: int = 0
    track_id: int = -1


# ============================================================
#  干扰物管理器
# ============================================================

class DistractorManager:
    """
    干扰物管理器 — 负责干扰物的检测、分类、跨帧跟踪与管理。

    生命周期:
        1. 跟踪初始化时创建实例
        2. 每帧调用 detect() 检测当前帧的干扰物
        3. 内部维护干扰物历史列表，支持跨帧关联
        4. 向 FusionModule 输出当前帧的干扰物列表

    参数:
        cfg (dict): 配置字典（distractor 节），包含各阈值参数
        device (str): 运行设备
    """

    def __init__(self, cfg: dict, device: str = "cpu"):
        distractor_cfg = cfg.get("distractor", {})

        # --- 检测参数 ---
        self.search_radius: int = distractor_cfg.get("search_radius", 80)
        self.attention_threshold: float = distractor_cfg.get("attention_threshold", 0.3)

        # --- 分类阈值 ---
        self.similarity_high: float = distractor_cfg.get("similarity_high", 0.6)
        self.dynamic_displacement: float = distractor_cfg.get("dynamic_displacement", 10.0)
        self.occlusion_overlap: float = distractor_cfg.get("occlusion_overlap", 0.3)

        # --- 特征权重 ---
        self.hog_weight: float = distractor_cfg.get("hog_weight", 0.4)
        self.cnn_weight: float = distractor_cfg.get("cnn_weight", 0.6)
        self.max_distractors: int = distractor_cfg.get("max_distractors", 10)

        # --- 内部状态 ---
        self.device = device
        self._current_distractors: List[DistractorInfo] = []   # 当前帧干扰物
        self._history: List[List[DistractorInfo]] = []          # 历史帧干扰物
        self._next_track_id: int = 0                            # 下一个可用跟踪 ID
        self._target_hog_feat: Optional[np.ndarray] = None      # 目标 HOG 特征缓存
        self._target_cnn_feat: Optional[torch.Tensor] = None    # 目标 CNN 特征缓存

        # --- ROI Align 参数 ---
        # roi_align 输出的空间尺寸，池化后再 flatten 得到与目标特征同维度的向量
        self._roi_output_size: int = distractor_cfg.get("roi_output_size", 7)
        # 当前帧的搜索区域裁剪几何信息（每帧由 Tracker 传入）
        self._search_region_info: Optional[dict] = None

        logger.info(
            f"DistractorManager 初始化完成 | "
            f"search_radius={self.search_radius}, "
            f"similarity_high={self.similarity_high}, "
            f"max_distractors={self.max_distractors}"
        )

    # ==========================================================
    #  公开接口
    # ==========================================================

    def initialize(
        self,
        target_patch: np.ndarray,
        target_cnn_feat: torch.Tensor,
    ) -> None:
        """
        跟踪初始化时调用，缓存目标的 HOG 和 CNN 特征作为参考。

        参数:
            target_patch (np.ndarray):     目标区域图像裁剪, shape = (H, W, 3), BGR
            target_cnn_feat (torch.Tensor): 目标的 CNN 深度特征向量
        """
        self._target_hog_feat = self._extract_hog_feature(target_patch)
        self._target_cnn_feat = target_cnn_feat.detach()
        self._current_distractors = []
        self._history = []
        self._next_track_id = 0

        logger.info("干扰物管理器已初始化，目标参考特征已缓存")

    def detect(
        self,
        frame: np.ndarray,
        target_bbox: np.ndarray,
        attention_map: Optional[torch.Tensor],
        search_cnn_features: Optional[torch.Tensor],
        frame_id: int,
        search_region_info: Optional[dict] = None,
    ) -> List[DistractorInfo]:
        """
        在当前帧中检测并分类干扰物（每帧调用一次）。

        完整流程:
          Step 1: 基于注意力图提取低关注度背景候选区域
          Step 2: 在候选区域中生成干扰物候选框
          Step 3: 计算每个候选框与目标的 HOG + CNN 特征相似度
          Step 4: 按阈值进行风险分级
          Step 5: 与历史帧干扰物进行跨帧关联
          Step 6: 按风险等级降序排列，保留 top-K

        参数:
            frame (np.ndarray):              当前帧图像, shape = (H, W, 3), BGR
            target_bbox (np.ndarray):        当前帧目标框 [x1, y1, x2, y2]
            attention_map (torch.Tensor):    CBAM 空间注意力图, shape = (1,1,H,W)
                                             None 表示 CBAM 未启用
            search_cnn_features (torch.Tensor): 搜索区域 CNN 特征图 (layer3),
                                                shape = (1, C, H_feat, W_feat)
            frame_id (int):                  当前帧号
            search_region_info (dict):       搜索区域裁剪几何信息（由 Tracker 传入）:
                "crop_x1":           裁剪原点 x（帧坐标）
                "crop_y1":           裁剪原点 y（帧坐标）
                "crop_size":         裁剪正方形边长（帧像素单位）
                "search_input_size": 模型输入尺寸（255）
                "feature_stride":    layer3 降采样率（16）

        返回:
            distractors (list[DistractorInfo]): 检测到的干扰物列表，按风险降序排列
        """
        # 缓存搜索区域几何信息，供 _cnn_similarity 做坐标映射
        self._search_region_info = search_region_info
        # Step 1: 提取低关注度候选区域
        candidate_regions = self._extract_candidate_regions(
            frame, target_bbox, attention_map
        )

        if len(candidate_regions) == 0:
            logger.debug(f"帧 {frame_id}: 未发现候选干扰区域")
            self._update_history([])
            return []

        # Step 2 & 3: 计算各候选区域的特征相似度
        distractors = []
        for region_bbox in candidate_regions:
            info = self._evaluate_candidate(
                frame, region_bbox, target_bbox, search_cnn_features, frame_id
            )
            if info is not None:
                distractors.append(info)

        # Step 4: 风险分级
        for d in distractors:
            d.risk_level = self._classify_risk(d)

        # Step 5: 跨帧关联
        self._associate_with_history(distractors)

        # Step 6: 排序 & 截断
        distractors.sort(key=lambda d: (d.risk_level, d.similarity), reverse=True)
        distractors = distractors[: self.max_distractors]

        # 更新内部状态
        self._current_distractors = distractors
        self._update_history(distractors)

        logger.debug(
            f"帧 {frame_id}: 检测到 {len(distractors)} 个干扰物 | "
            f"风险分布: {self._risk_summary(distractors)}"
        )

        return distractors

    def get_current_distractors(self) -> List[DistractorInfo]:
        """获取当前帧的干扰物列表（只读）。"""
        return self._current_distractors.copy()

    def get_high_risk_distractors(self) -> List[DistractorInfo]:
        """获取当前帧中风险等级 ≥ OCCLUSION 的干扰物。"""
        return [
            d for d in self._current_distractors
            if d.risk_level >= RiskLevel.OCCLUSION
        ]

    # ==========================================================
    #  内部方法 — 候选区域提取
    # ==========================================================

    def _extract_candidate_regions(
        self,
        frame: np.ndarray,
        target_bbox: np.ndarray,
        attention_map: Optional[torch.Tensor],
    ) -> List[np.ndarray]:
        """
        基于 CBAM 注意力图提取低关注度的背景候选区域。

        逻辑:
          1. 将注意力图二值化（< threshold 的区域视为背景）
          2. 在背景区域中寻找连通分量
          3. 对连通分量取外接矩形作为候选框
          4. 排除与目标框高度重叠的区域（避免将目标自身误判为干扰物）

        参数:
            frame:         当前帧图像
            target_bbox:   目标框 [x1, y1, x2, y2]
            attention_map: CBAM 空间注意力图 (可选)

        返回:
            candidates: 候选区域框列表, 每个元素为 [x1, y1, x2, y2]
        """
        H, W = frame.shape[:2]
        candidates = []

        if attention_map is not None:
            # --- 基于注意力图的候选区域提取 ---
            attn = attention_map.squeeze().cpu().numpy()  # (h, w)

            # 上采样到原图尺寸
            attn_resized = cv2.resize(attn, (W, H), interpolation=cv2.INTER_LINEAR)

            # 二值化：低关注度区域 = 背景
            bg_mask = (attn_resized < self.attention_threshold).astype(np.uint8) * 255

            # 寻找连通分量
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
                bg_mask, connectivity=8
            )

            # 遍历连通分量（跳过 label=0 即背景）
            for i in range(1, num_labels):
                x, y, w, h, area = stats[i]
                # 过滤面积过小或过大的区域
                if area < 100 or area > H * W * 0.5:
                    continue
                candidate = np.array([x, y, x + w, y + h])

                # 排除与目标框重叠度过高的区域
                if self._compute_iou(candidate, target_bbox) < 0.7:
                    candidates.append(candidate)
        else:
            # --- 注意力图不可用时的回退方案：滑动窗口 ---
            candidates = self._sliding_window_candidates(
                frame, target_bbox, self.search_radius
            )

        return candidates

    def _sliding_window_candidates(
        self,
        frame: np.ndarray,
        target_bbox: np.ndarray,
        radius: int,
    ) -> List[np.ndarray]:
        """
        滑动窗口生成候选区域（CBAM 不可用时的回退方案）。

        在目标框周围 radius 范围内，以目标框大小为基准，
        按一定步长滑动生成候选框。
        """
        H, W = frame.shape[:2]
        tx1, ty1, tx2, ty2 = target_bbox.astype(int)
        tw, th = tx2 - tx1, ty2 - ty1
        step = max(tw, th) // 2  # 步长为目标尺寸的一半

        candidates = []
        cx, cy = (tx1 + tx2) // 2, (ty1 + ty2) // 2

        for dy in range(-radius, radius + 1, step):
            for dx in range(-radius, radius + 1, step):
                nx, ny = cx + dx, cy + dy
                x1 = max(0, nx - tw // 2)
                y1 = max(0, ny - th // 2)
                x2 = min(W, x1 + tw)
                y2 = min(H, y1 + th)
                candidate = np.array([x1, y1, x2, y2])
                if self._compute_iou(candidate, target_bbox) < 0.3:
                    candidates.append(candidate)

        return candidates

    # ==========================================================
    #  内部方法 — 特征计算与相似度
    # ==========================================================

    def _evaluate_candidate(
        self,
        frame: np.ndarray,
        candidate_bbox: np.ndarray,
        target_bbox: np.ndarray,
        search_cnn_features: Optional[torch.Tensor],
        frame_id: int,
    ) -> Optional[DistractorInfo]:
        """
        评估单个候选区域：计算特征相似度，生成 DistractorInfo。

        参数:
            frame:              当前帧图像
            candidate_bbox:     候选区域框 [x1, y1, x2, y2]
            target_bbox:        目标框
            search_cnn_features: 搜索区域 CNN 特征
            frame_id:           当前帧号

        返回:
            DistractorInfo 或 None（若相似度过低则返回 None）
        """
        x1, y1, x2, y2 = candidate_bbox.astype(int)
        patch = frame[y1:y2, x1:x2]

        if patch.size == 0:
            return None

        # --- HOG 特征相似度 ---
        hog_feat = self._extract_hog_feature(patch)
        hog_sim = self._cosine_similarity(hog_feat, self._target_hog_feat)

        # --- CNN 特征相似度 ---
        cnn_sim = 0.0
        if search_cnn_features is not None and self._target_cnn_feat is not None:
            cnn_sim = self._cnn_similarity(search_cnn_features, candidate_bbox)

        # --- 加权融合相似度 ---
        combined_sim = self.hog_weight * hog_sim + self.cnn_weight * cnn_sim

        # 低相似度候选直接丢弃（节省计算量）
        if combined_sim < 0.2:
            return None

        # --- 计算与目标的 IoU ---
        iou = self._compute_iou(candidate_bbox, target_bbox)

        # --- 计算帧间位移（与历史干扰物对比）---
        displacement = self._estimate_displacement(candidate_bbox)

        return DistractorInfo(
            bbox=candidate_bbox,
            similarity=combined_sim,
            hog_similarity=hog_sim,
            cnn_similarity=cnn_sim,
            displacement=displacement,
            iou_with_target=iou,
            frame_id=frame_id,
        )

    def _extract_hog_feature(self, patch: np.ndarray) -> np.ndarray:
        """
        提取图像区域的 HOG 特征描述子。

        对应开题报告：利用 HOG 特征捕捉目标与干扰物的形状细节。

        参数:
            patch: 图像区域, shape = (H, W, 3), BGR

        返回:
            hog_descriptor: HOG 特征向量, shape = (N,)
        """
        # 统一缩放到固定尺寸，保证特征维度一致
        resized = cv2.resize(patch, (64, 128))
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

        # OpenCV HOG 描述子
        win_size = (64, 128)
        block_size = (16, 16)
        block_stride = (8, 8)
        cell_size = (8, 8)
        nbins = 9

        hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, nbins)
        descriptor = hog.compute(gray)

        if descriptor is None:
            return np.zeros(3780)  # HOG 默认输出维度

        return descriptor.flatten()

    def _cosine_similarity(self, feat_a: np.ndarray, feat_b: np.ndarray) -> float:
        """计算两个特征向量的余弦相似度，返回值域 [0, 1]。"""
        if feat_a is None or feat_b is None:
            return 0.0
        norm_a = np.linalg.norm(feat_a)
        norm_b = np.linalg.norm(feat_b)
        if norm_a < 1e-8 or norm_b < 1e-8:
            return 0.0
        return float(np.clip(np.dot(feat_a, feat_b) / (norm_a * norm_b), 0.0, 1.0))

    def _cnn_similarity(
        self,
        search_features: torch.Tensor,
        candidate_bbox: np.ndarray,
    ) -> float:
        """
        基于 CNN 深度特征计算候选区域与目标的余弦相似度。

        核心思路:
          1. 将 candidate_bbox 从原始帧像素坐标映射到 CNN 特征图坐标
          2. 使用 torchvision.ops.roi_align 从特征图中精确提取候选区域特征
          3. 全局平均池化得到 1D 特征向量
          4. 与目标参考特征计算余弦相似度

        坐标映射链路:
          candidate_bbox [x1,y1,x2,y2] (帧像素坐标)
              │  步骤1: 减去裁剪原点 (crop_x1, crop_y1)
              ▼
          搜索裁剪坐标 (crop_size × crop_size 区域内)
              │  步骤2: 乘以 (search_input_size / crop_size)
              ▼
          模型输入坐标 (255 × 255 图像内)
              │  步骤3: 除以 feature_stride (16)
              ▼
          特征图坐标 (H_feat × W_feat) → 传入 roi_align

        参数:
            search_features (torch.Tensor): 搜索区域的 layer3 特征图,
                                            shape = (1, C, H_feat, W_feat)
            candidate_bbox (np.ndarray):    候选框 [x1, y1, x2, y2]（原始帧像素坐标）

        返回:
            similarity (float): 余弦相似度 [0, 1]，越接近 1 越相似
        """
        if self._target_cnn_feat is None:
            return 0.0

        # ---- 安全检查: 搜索区域几何信息是否可用 ----
        if self._search_region_info is None:
            logger.warning(
                "_cnn_similarity: search_region_info 为 None，"
                "回退到全局平均池化"
            )
            return self._cnn_similarity_fallback(search_features)

        info = self._search_region_info
        crop_x1 = info["crop_x1"]
        crop_y1 = info["crop_y1"]
        crop_size = info["crop_size"]
        input_size = info["search_input_size"]   # 255
        stride = info["feature_stride"]          # 16

        # ---- 步骤 1: 帧坐标 → 搜索裁剪坐标 ----
        # 减去裁剪区域的左上角原点
        x1_crop = candidate_bbox[0] - crop_x1
        y1_crop = candidate_bbox[1] - crop_y1
        x2_crop = candidate_bbox[2] - crop_x1
        y2_crop = candidate_bbox[3] - crop_y1

        # ---- 步骤 2: 搜索裁剪坐标 → 模型输入坐标 (255×255) ----
        # crop_size 的正方形区域被 resize 到 input_size，需乘以缩放比
        scale = input_size / max(crop_size, 1e-6)
        x1_input = x1_crop * scale
        y1_input = y1_crop * scale
        x2_input = x2_crop * scale
        y2_input = y2_crop * scale

        # ---- 步骤 3: 模型输入坐标 → 特征图坐标 ----
        # layer3 的空间降采样率为 stride
        x1_feat = x1_input / stride
        y1_feat = y1_input / stride
        x2_feat = x2_input / stride
        y2_feat = y2_input / stride

        # ---- 边界合法性检查 ----
        _, _, H_feat, W_feat = search_features.shape
        x1_feat = max(0.0, x1_feat)
        y1_feat = max(0.0, y1_feat)
        x2_feat = min(float(W_feat), x2_feat)
        y2_feat = min(float(H_feat), y2_feat)

        # 候选框在特征图上面积过小则跳过（至少 1×1 的特征区域）
        if (x2_feat - x1_feat) < 0.5 or (y2_feat - y1_feat) < 0.5:
            logger.debug(
                f"候选框映射到特征图后面积过小: "
                f"({x1_feat:.1f},{y1_feat:.1f})-({x2_feat:.1f},{y2_feat:.1f}), 跳过"
            )
            return 0.0

        # ---- 步骤 4: roi_align 提取候选区域特征 ----
        # roi_align 要求 rois 格式: (batch_index, x1, y1, x2, y2)
        rois = torch.tensor(
            [[0, x1_feat, y1_feat, x2_feat, y2_feat]],
            dtype=torch.float32,
            device=search_features.device,
        )

        # roi_align 输出: (1, C, roi_output_size, roi_output_size)
        roi_feat = torchvision.ops.roi_align(
            input=search_features,
            boxes=rois,
            output_size=self._roi_output_size,   # 默认 7×7
            spatial_scale=1.0,   # rois 已经是特征图坐标，无需额外缩放
            aligned=True,        # PyTorch ≥1.5 推荐开启，修正半像素偏移
        )

        # 全局平均池化 → 1D 特征向量, shape = (C,)
        candidate_feat = F.adaptive_avg_pool2d(roi_feat, 1).flatten()

        # ---- 步骤 5: 余弦相似度 ----
        target_feat = self._target_cnn_feat.to(candidate_feat.device)

        cosine_sim = F.cosine_similarity(
            candidate_feat.unsqueeze(0),
            target_feat.unsqueeze(0),
        ).item()

        # 将 [-1, 1] 的余弦相似度映射到 [0, 1]
        similarity = float(np.clip((cosine_sim + 1.0) / 2.0, 0.0, 1.0))

        return similarity

    def _cnn_similarity_fallback(
        self, search_features: torch.Tensor
    ) -> float:
        """
        回退方案: 当 search_region_info 不可用时，
        使用全局平均池化计算粗略相似度（精度较低）。
        """
        candidate_feat = F.adaptive_avg_pool2d(search_features, 1).flatten()
        target_feat = self._target_cnn_feat.to(candidate_feat.device)

        cosine_sim = F.cosine_similarity(
            candidate_feat.unsqueeze(0),
            target_feat.unsqueeze(0),
        ).item()

        return float(np.clip((cosine_sim + 1.0) / 2.0, 0.0, 1.0))

    # ==========================================================
    #  内部方法 — 风险分级
    # ==========================================================

    def _classify_risk(self, info: DistractorInfo) -> RiskLevel:
        """
        根据干扰物特征将其划分为对应风险等级。

        分类规则：
          - 高相似干扰物: similarity ≥ similarity_high
          - 动态干扰物:   displacement ≥ dynamic_displacement
          - 遮挡型干扰物: iou_with_target ≥ occlusion_overlap
          - 若同时满足多个条件，取最高风险等级

        参数:
            info: 干扰物信息

        返回:
            RiskLevel 枚举值
        """
        level = RiskLevel.LOW

        # 动态干扰物判定
        if info.displacement >= self.dynamic_displacement:
            level = max(level, RiskLevel.DYNAMIC)

        # 遮挡型干扰物判定
        if info.iou_with_target >= self.occlusion_overlap:
            level = max(level, RiskLevel.OCCLUSION)

        # 高相似干扰物判定（最高风险）
        if info.similarity >= self.similarity_high:
            level = max(level, RiskLevel.HIGH_SIMILAR)

        return level

    # ==========================================================
    #  内部方法 — 跨帧关联
    # ==========================================================

    def _associate_with_history(self, distractors: List[DistractorInfo]) -> None:
        """
        将当前帧干扰物与历史帧干扰物进行跨帧关联（简单 IoU 匹配）。

        匹配成功则继承 track_id，否则分配新 ID。
        """
        if not self._history:
            for d in distractors:
                d.track_id = self._next_track_id
                self._next_track_id += 1
            return

        prev_distractors = self._history[-1]

        for d in distractors:
            best_iou = 0.0
            best_id = -1
            for pd in prev_distractors:
                iou = self._compute_iou(d.bbox, pd.bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_id = pd.track_id

            if best_iou > 0.3 and best_id >= 0:
                d.track_id = best_id
            else:
                d.track_id = self._next_track_id
                self._next_track_id += 1

    def _estimate_displacement(self, bbox: np.ndarray) -> float:
        """估算候选区域的帧间位移（与上一帧最近干扰物的中心距离）。"""
        if not self._history:
            return 0.0

        cx, cy = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
        min_dist = float("inf")

        for pd in self._history[-1]:
            pcx = (pd.bbox[0] + pd.bbox[2]) / 2
            pcy = (pd.bbox[1] + pd.bbox[3]) / 2
            dist = np.sqrt((cx - pcx) ** 2 + (cy - pcy) ** 2)
            min_dist = min(min_dist, dist)

        return min_dist if min_dist != float("inf") else 0.0

    def _update_history(self, distractors: List[DistractorInfo]) -> None:
        """更新历史记录（保留最近 10 帧）。"""
        self._history.append(distractors)
        if len(self._history) > 10:
            self._history.pop(0)

    # ==========================================================
    #  工具方法
    # ==========================================================

    @staticmethod
    def _compute_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
        """计算两个框的 IoU (Intersection over Union)。"""
        x1 = max(box_a[0], box_b[0])
        y1 = max(box_a[1], box_b[1])
        x2 = min(box_a[2], box_b[2])
        y2 = min(box_a[3], box_b[3])

        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
        area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
        union = area_a + area_b - inter

        return inter / union if union > 0 else 0.0

    @staticmethod
    def _risk_summary(distractors: List[DistractorInfo]) -> str:
        """生成风险等级分布的简要统计字符串。"""
        from collections import Counter
        counter = Counter(d.risk_level.name for d in distractors)
        return ", ".join(f"{k}={v}" for k, v in counter.items())