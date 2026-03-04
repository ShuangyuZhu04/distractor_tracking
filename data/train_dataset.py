"""
train_dataset.py — 训练数据集 (TrackingTrainDataset)
=======================================================
组合序列数据、数据增强、anchor 标签分配、干扰物掩码生成，
输出可直接喂给 DataLoader 的 (template, search, labels) 元组。

__getitem__ 输出:
    template:        (3, 127, 127)  float32 张量
    search:          (3, 255, 255)  float32 张量
    cls_label:       (A*S*S,)       int64   分类标签 {-1, 0, 1}
    reg_label:       (4, A*S*S)     float32 回归目标
    distractor_mask: (A*S*S,)       float32 干扰物掩码 {0, 1}

训练帧对采样策略:
    1. 随机选一个序列
    2. 随机选一帧作为模板帧
    3. 在 ±max_frame_gap 范围内随机选搜索帧
    4. 对搜索区域施加随机平移 + 缩放增强
"""

import logging
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .anchor import AnchorTargetGenerator
from .datasets import BaseTrackingDataset, SequenceInfo

logger = logging.getLogger(__name__)


# ============================================================
#  ImageNet 归一化常量
# ============================================================
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class TrackingTrainDataset(Dataset):
    """
    SiamRPN++ 训练数据集。

    生命周期:
        1. 初始化时加载所有序列元信息
        2. 每个 epoch 随机采样 (模板帧, 搜索帧) 对
        3. __getitem__ 执行裁剪、增强、标签生成

    参数:
        sequences (list[SequenceInfo]):  序列列表（来自 BaseTrackingDataset）
        cfg (dict):                      完整配置字典
    """

    def __init__(self, sequences: List[SequenceInfo], cfg: dict):
        super().__init__()
        self.sequences = sequences
        self.cfg = cfg

        # ---- 裁剪参数 ----
        tracker_cfg = cfg["tracker"]
        self.template_size: int = tracker_cfg["template_size"]    # 127
        self.search_size: int = tracker_cfg["search_size"]        # 255
        self.context_amount: float = tracker_cfg["context_amount"]

        # ---- 采样参数 ----
        train_cfg = cfg["train"]
        self.max_frame_gap: int = train_cfg.get("max_frame_gap", 100)
        self.pairs_per_seq: int = train_cfg.get("pairs_per_sequence", 2)

        # ---- 增强参数 ----
        #   shift 改为裁剪尺寸的比例而非固定像素
        #   shift_ratio=0.25 → 目标在 255×255 搜索图中偏移 ±64px
        self.shift_ratio: float = train_cfg.get("shift_ratio", 0.25)
        self.scale_range: Tuple[float, float] = (0.92, 1.08)

        # ---- Anchor 目标生成器 ----
        rpn_cfg = cfg["rpn"]
        self.anchor_gen = AnchorTargetGenerator(
            score_size=train_cfg.get("score_size", 17),
            stride=tracker_cfg.get("stride", 8),
            search_size=self.search_size,
            anchor_scales=rpn_cfg["anchor_scales"],
            anchor_ratios=rpn_cfg["anchor_ratios"],
            pos_iou_thr=train_cfg.get("pos_iou_thr", 0.6),
            neg_iou_thr=train_cfg.get("neg_iou_thr", 0.3),
            total_sample_num=train_cfg.get("anchor_sample_num", 64),
            pos_ratio=train_cfg.get("anchor_pos_ratio", 0.25),
        )

        # ---- 干扰物掩码参数 ----
        # IoU 在 [neg_thr, pos_thr) 之间的 anchor 被视为「潜在干扰物」
        # 这些 anchor 既不够正也不够负，是最容易迷惑模型的位置
        self._dist_iou_low: float = train_cfg.get("neg_iou_thr", 0.3)
        self._dist_iou_high: float = train_cfg.get("pos_iou_thr", 0.6)
        # 随机采样的干扰物 anchor 数量上限
        self._max_distractor_anchors: int = train_cfg.get("max_distractor_anchors", 16)

        # ---- 构建帧对索引 ----
        self._pairs = self._build_pair_list()

        logger.info(
            f"TrackingTrainDataset 初始化完成 | "
            f"序列数={len(sequences)}, "
            f"帧对数={len(self._pairs)}, "
            f"total_anchors={self.anchor_gen.total_anchors}, "
            f"shift_ratio={self.shift_ratio}"
        )

    # ==========================================================
    #  帧对索引构建
    # ==========================================================

    def _build_pair_list(self) -> List[Tuple[int, int, int]]:
        """
        预构建 (seq_idx, template_frame_idx, search_frame_idx) 列表。

        每个序列生成 pairs_per_seq 个帧对。
        """
        pairs = []
        for seq_idx, seq in enumerate(self.sequences):
            n_frames = len(seq)
            if n_frames < 2:
                continue

            for _ in range(self.pairs_per_seq):
                # 随机选模板帧
                t_idx = np.random.randint(0, n_frames)

                # 在 ±max_frame_gap 范围内选搜索帧（不同于模板帧）
                low = max(0, t_idx - self.max_frame_gap)
                high = min(n_frames, t_idx + self.max_frame_gap + 1)
                s_idx = t_idx
                while s_idx == t_idx:
                    s_idx = np.random.randint(low, high)

                pairs.append((seq_idx, t_idx, s_idx))

        return pairs

    def reshuffle(self):
        """每个 epoch 开始时重新生成帧对列表（保证随机性）。"""
        self._pairs = self._build_pair_list()

    def __len__(self) -> int:
        return len(self._pairs)

    # ==========================================================
    #  核心: __getitem__
    # ==========================================================

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        """
        采样一对 (模板, 搜索) 图像并生成全部训练标签。

        返回:
            dict:
                "template":        (3, 127, 127)  float32
                "search":          (3, 255, 255)  float32
                "cls_label":       (total_anchors,)   int64
                "reg_label":       (4, total_anchors)  float32
                "distractor_mask": (total_anchors,)   float32
        """
        seq_idx, t_idx, s_idx = self._pairs[index]
        seq = self.sequences[seq_idx]

        # ---- 读取图像 ----
        template_img = cv2.imread(seq.frame_paths[t_idx])
        search_img = cv2.imread(seq.frame_paths[s_idx])

        # 读取失败时回退到随机样本
        if template_img is None or search_img is None:
            return self.__getitem__(np.random.randint(len(self)))

        # ---- 读取 GT bbox [x, y, w, h] ----
        t_gt = seq.ground_truth[t_idx].copy()
        s_gt = seq.ground_truth[s_idx].copy()

        # 跳过无效标注
        if t_gt[2] <= 0 or t_gt[3] <= 0 or s_gt[2] <= 0 or s_gt[3] <= 0:
            return self.__getitem__(np.random.randint(len(self)))

        # ---- 裁剪模板 (无增强) ----
        template_patch = self._crop_and_resize(
            template_img, t_gt, self.template_size
        )

        # ---- 裁剪搜索 (带随机增强) + 计算 GT 在裁剪图中的坐标 ----
        search_patch, gt_in_search = self._crop_search_with_gt(
            search_img, s_gt
        )

        # ---- 颜色增强 + 模糊/灰度增强 ----
        template_patch = self._color_jitter(template_patch, amount=0.15)
        search_patch = self._color_jitter(search_patch, amount=0.15)
        template_patch = self._random_blur(template_patch)
        search_patch = self._random_blur(search_patch)
        template_patch = self._random_grayscale(template_patch)
        search_patch = self._random_grayscale(search_patch)

        # ---- 随机水平翻转（模板和搜索同步）----
        if np.random.random() < 0.5:
            template_patch = np.ascontiguousarray(template_patch[:, ::-1])
            search_patch = np.ascontiguousarray(search_patch[:, ::-1])
            # 翻转 GT: cx 镜像
            gt_in_search[0] = self.search_size - 1 - gt_in_search[0]

        # ---- 转张量 + 归一化 ----
        template_tensor = self._to_tensor(template_patch)
        search_tensor = self._to_tensor(search_patch)

        # ---- 生成 anchor 标签 ----
        anchor_targets = self.anchor_gen.generate(gt_in_search)

        # ---- 生成干扰物掩码 ----
        distractor_mask = self._generate_distractor_mask(gt_in_search)

        #   激活干扰物 anchor 为困难负样本
        #   IoU [0.3, 0.6) 的 anchor 原本是 ignore (label=-1),
        #   现在被 distractor_mask 选中后激活为 negative (label=0),
        #   配合 dist_loss 高权重, 迫使模型区分目标与困难干扰物
        cls_label = anchor_targets["cls_label"].copy()
        dist_activate = (distractor_mask > 0) & (cls_label == -1)
        cls_label[dist_activate] = 0  # ignore -> hard negative

        return {
            "template": template_tensor,                                          # (3, 127, 127)
            "search": search_tensor,                                              # (3, 255, 255)
            "cls_label": torch.from_numpy(cls_label),                             # (N,) int64
            "reg_label": torch.from_numpy(
                anchor_targets["reg_label"].T.astype(np.float32)                  # (4, N) float32
            ),
            "distractor_mask": torch.from_numpy(distractor_mask),                 # (N,) float32
        }

    # ==========================================================
    #  裁剪与增强
    # ==========================================================

    def _crop_and_resize(
        self, img: np.ndarray, bbox_xywh: np.ndarray, output_size: int
    ) -> np.ndarray:
        """以 bbox 中心裁剪正方形区域并缩放。与 Tracker._crop_patch 逻辑一致。"""
        x, y, w, h = bbox_xywh
        cx, cy = x + w / 2, y + h / 2

        context = self.context_amount * (w + h)
        crop_size = int(np.sqrt((w + context) * (h + context)))
        crop_size = max(crop_size, 1)

        x1 = int(cx - crop_size / 2)
        y1 = int(cy - crop_size / 2)
        x2 = x1 + crop_size
        y2 = y1 + crop_size

        H, W = img.shape[:2]
        pad = [max(0, -x1), max(0, -y1), max(0, x2 - W), max(0, y2 - H)]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(W, x2), min(H, y2)

        patch = img[y1:y2, x1:x2]
        if any(pad):
            patch = cv2.copyMakeBorder(
                patch, pad[1], pad[3], pad[0], pad[2],
                cv2.BORDER_CONSTANT, value=(0, 0, 0),
            )

        return cv2.resize(patch, (output_size, output_size))

    def _crop_search_with_gt(
        self, img: np.ndarray, bbox_xywh: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        裁剪搜索区域，同时计算 GT bbox 在裁剪图中的坐标。

        关键公式 (SiamRPN++ 标准做法):
          exemplar_crop = sqrt((w + context) * (h + context))
          search_crop   = exemplar_crop × (search_size / template_size)
                        = exemplar_crop × (255 / 127) ≈ exemplar_crop × 2.0

        这样目标在搜索图 (255×255) 中的像素尺度与模板图 (127×127) 中
        近似相同（都约 50~70px），从而与 anchor base_size=64 匹配。

        增强: 对裁剪中心施加与 base_crop_size 成正比的随机平移。
        shift_ratio × base_crop_size, 在 255×255 搜索图中产生约 ±(shift_ratio × 255) ≈ ±64px 的目标偏移,
        迫使模型学习真正的特征匹配。

        返回:
            search_patch: (255, 255, 3)
            gt_in_search: [cx, cy, w, h] 在 255×255 搜索图中的坐标
        """
        x, y, w, h = bbox_xywh
        cx, cy = x + w / 2, y + h / 2

        context = self.context_amount * (w + h)
        exemplar_crop = np.sqrt((w + context) * (h + context))

        # 搜索裁剪范围 = 模板裁剪 × (search_size / template_size)
        # 这保证目标在搜索图中保持与模板相同的像素尺度
        base_crop_size = exemplar_crop * (self.search_size / self.template_size)

        # ---- 随机增强 ----
        #   平移量与裁剪尺寸成正比 
        #   shift_ratio=0.25 → 搜索图中目标偏移 ≈ ±0.25*255 ≈ ±64px
        max_shift = self.shift_ratio * base_crop_size
        dx = np.random.uniform(-max_shift, max_shift)
        dy = np.random.uniform(-max_shift, max_shift)
        scale_factor = np.random.uniform(*self.scale_range)

        crop_size = base_crop_size * scale_factor
        crop_size = max(crop_size, 1.0)

        crop_cx = cx + dx
        crop_cy = cy + dy

        # ---- 裁剪 ----
        crop_x1 = int(crop_cx - crop_size / 2)
        crop_y1 = int(crop_cy - crop_size / 2)
        crop_x2 = crop_x1 + int(crop_size)
        crop_y2 = crop_y1 + int(crop_size)

        H, W = img.shape[:2]
        pad = [max(0, -crop_x1), max(0, -crop_y1),
               max(0, crop_x2 - W), max(0, crop_y2 - H)]

        crop_x1_clipped = max(0, crop_x1)
        crop_y1_clipped = max(0, crop_y1)
        crop_x2_clipped = min(W, crop_x2)
        crop_y2_clipped = min(H, crop_y2)

        patch = img[crop_y1_clipped:crop_y2_clipped, crop_x1_clipped:crop_x2_clipped]
        if any(pad):
            patch = cv2.copyMakeBorder(
                patch, pad[1], pad[3], pad[0], pad[2],
                cv2.BORDER_CONSTANT, value=(0, 0, 0),
            )

        search_patch = cv2.resize(patch, (self.search_size, self.search_size))

        # ---- 计算 GT 在 255×255 搜索图中的坐标 ----
        resize_scale = self.search_size / crop_size

        gt_cx_in_search = (cx - crop_x1) * resize_scale
        gt_cy_in_search = (cy - crop_y1) * resize_scale
        gt_w_in_search = w * resize_scale
        gt_h_in_search = h * resize_scale

        gt_in_search = np.array(
            [gt_cx_in_search, gt_cy_in_search, gt_w_in_search, gt_h_in_search],
            dtype=np.float64,
        )

        return search_patch, gt_in_search

    def _color_jitter(self, img: np.ndarray, amount: float = 0.1) -> np.ndarray:
        """亮度 + 对比度随机扰动。"""
        img = img.astype(np.float32)

        brightness = 1.0 + np.random.uniform(-amount, amount)
        img *= brightness

        contrast = 1.0 + np.random.uniform(-amount, amount)
        mean = img.mean()
        img = (img - mean) * contrast + mean

        return np.clip(img, 0, 255).astype(np.uint8)

    @staticmethod
    def _to_tensor(img: np.ndarray) -> torch.Tensor:
        """BGR uint8 图像 → 归一化 float32 张量, shape (3, H, W)。"""
        # BGR → RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        # ImageNet 归一化
        img = (img - IMAGENET_MEAN) / IMAGENET_STD
        # HWC → CHW
        return torch.from_numpy(img.transpose(2, 0, 1)).float()

    # ==========================================================
    #  干扰物掩码生成（核心创新点）
    # ==========================================================

    def _generate_distractor_mask(
        self, gt_in_search: np.ndarray
    ) -> np.ndarray:
        """
        生成干扰物训练掩码 -- 挖掘困难负样本。

        策略:
          1. 主力: IoU [neg_thr, pos_thr) 即 [0.3, 0.6) 的困难负样本
          2. 补充: IoU [0.1, neg_thr) 且尺度相似的中等干扰物

        返回:
            mask: (total_anchors,) float32, 1=干扰物, 0=普通
        """
        anchors = self.anchor_gen.anchors
        total = self.anchor_gen.total_anchors
        mask = np.zeros(total, dtype=np.float32)

        gt = gt_in_search.astype(np.float64)
        ious = self.anchor_gen._compute_iou(anchors, gt)

        # 1. 主力干扰物: IoU [neg_thr, pos_thr) -- 困难负样本
        hard_idx = np.where(
            (ious >= self._dist_iou_low) & (ious < self._dist_iou_high)
        )[0]

        # 2. 补充干扰物: IoU [0.1, neg_thr) 且尺度相似
        gt_w, gt_h = max(gt[2], 1e-8), max(gt[3], 1e-8)
        scale_w = anchors[:, 2] / gt_w
        scale_h = anchors[:, 3] / gt_h
        similar_scale = (
            (scale_w >= 0.5) & (scale_w <= 2.0) &
            (scale_h >= 0.5) & (scale_h <= 2.0)
        )
        medium_idx = np.where(
            (ious >= 0.1) & (ious < self._dist_iou_low) & similar_scale
        )[0]

        # 3. 采样: 优先困难, 补充中等
        n_hard = min(len(hard_idx), self._max_distractor_anchors)
        if n_hard > 0:
            sel = np.random.choice(hard_idx, size=n_hard, replace=False)
            mask[sel] = 1.0

        remaining = self._max_distractor_anchors - n_hard
        if remaining > 0 and len(medium_idx) > 0:
            n_med = min(len(medium_idx), remaining)
            sel = np.random.choice(medium_idx, size=n_med, replace=False)
            mask[sel] = 1.0

        return mask

    #  工厂方法
    # ==========================================================

    @staticmethod
    def _random_blur(img: np.ndarray, prob: float = 0.15) -> np.ndarray:
        """以一定概率施加高斯模糊, 增加训练鲁棒性。"""
        if np.random.random() < prob:
            ksize = np.random.choice([3, 5])
            img = cv2.GaussianBlur(img, (ksize, ksize), 0)
        return img

    @staticmethod
    def _random_grayscale(img: np.ndarray, prob: float = 0.05) -> np.ndarray:
        """以一定概率转灰度图 (保留3通道), 减少颜色过拟合。"""
        if np.random.random() < prob:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.merge([gray, gray, gray])
        return img

    @classmethod
    def from_config(
        cls,
        cfg: dict,
        dataset_instances: List[BaseTrackingDataset],
    ) -> "TrackingTrainDataset":
        """
        从配置和数据集实例列表构建训练数据集。

        参数:
            cfg:                完整配置字典
            dataset_instances:  已加载的 BaseTrackingDataset 列表

        返回:
            TrackingTrainDataset 实例
        """
        all_sequences = []
        for ds in dataset_instances:
            all_sequences.extend(ds.sequences)

        logger.info(
            f"合并 {len(dataset_instances)} 个数据集, "
            f"共 {len(all_sequences)} 个序列"
        )

        return cls(all_sequences, cfg)