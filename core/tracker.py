"""
tracker.py — 跟踪主控器 (Tracker)
====================================
统一协调模型推理、干扰物挖掘、融合优化的完整跟踪流程。

单帧跟踪流程:
  ┌──────────────────────────────────────────────────────────┐
  │  输入: 当前帧图像 frame, 上一帧目标框 prev_bbox         │
  │                                                          │
  │  1. 裁剪搜索区域 (基于 prev_bbox + context)              │
  │  2. SiamRPN++ 前向推理 → cls_score, bbox_pred,          │
  │                           attention_map                  │
  │  3. 解码得到候选框 + 置信度                              │
  │  4. DistractorManager.detect() → 干扰物列表              │
  │  5. FusionModule.compute_adjustment() → 调整指令         │
  │  6. 应用调整 → 输出最终 target_bbox, score               │
  │                                                          │
  │  输出: 当前帧目标框 target_bbox, 置信度 score,           │
  │        干扰物列表 distractors                            │
  └──────────────────────────────────────────────────────────┘
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import yaml

from models.siamrpn import SiamRPNPP
from .distractor import DistractorInfo, DistractorManager
from .fusion import FusionModule, TrackingAdjustment
from data.anchor import AnchorTargetGenerator

logger = logging.getLogger(__name__)


class Tracker:
    """
    跟踪主控器 — 封装从初始化到逐帧跟踪的全部逻辑。

    使用示例:
        >>> cfg = yaml.safe_load(open("configs/default.yaml"))
        >>> tracker = Tracker(cfg)
        >>> tracker.initialize(first_frame, init_bbox)
        >>> for frame in video_frames:
        ...     result = tracker.track(frame)
        ...     print(result["bbox"], result["score"])

    参数:
        cfg (dict): 完整配置字典
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.device = cfg["system"]["device"]

        # ---- 模型 ----
        self.model = SiamRPNPP(cfg).to(self.device)

        # ---- 干扰物管理器 ----
        distractor_cfg = cfg.get("distractor", {})
        self.use_distractor = distractor_cfg.get("enabled", True)
        if self.use_distractor:
            self.distractor_mgr = DistractorManager(cfg, self.device)
        else:
            self.distractor_mgr = None

        # ---- 融合优化模块 ----
        fusion_cfg = cfg.get("fusion", {})
        self.use_fusion = fusion_cfg.get("enabled", True)
        if self.use_fusion:
            self.fusion_module = FusionModule(cfg)
        else:
            self.fusion_module = None

        # ---- 跟踪参数 ----
        tracker_cfg = cfg["tracker"]
        self.template_size: int = tracker_cfg["template_size"]   # 127
        self.search_size: int = tracker_cfg["search_size"]       # 255
        self.context_amount: float = tracker_cfg["context_amount"]
        self.score_threshold: float = tracker_cfg["score_threshold"]
        self.window_influence: float = tracker_cfg["window_influence"]
        self.scale_lr: float = tracker_cfg["lr"]
        self.penalty_k: float = tracker_cfg.get("penalty_k", 0.055)

        # layer3 的空间降采样率（ResNet50: layer3 stride=16）
        # 用于将原图像素坐标映射到特征图坐标，供干扰物模块 ROI Align 使用
        self.feature_stride: int = tracker_cfg.get("feature_stride", 16)

        # 初始化 Anchor 生成器，获取先验框
        self.anchor_gen = AnchorTargetGenerator(
            score_size=cfg["train"].get("score_size", 17),
            stride=tracker_cfg.get("stride", 8),
            search_size=self.search_size,
            anchor_scales=cfg["rpn"]["anchor_scales"],
            anchor_ratios=cfg["rpn"]["anchor_ratios"],
        )
        self.anchors = self.anchor_gen.anchors  # shape: (1445, 4) [cx, cy, w, h]

        # ---- 内部跟踪状态 ----
        self._target_bbox: Optional[np.ndarray] = None   # [cx, cy, w, h]
        self._frame_id: int = 0
        self._is_initialized: bool = False

        # 余弦窗（用于跟踪框位置平滑）
        self._window = self._create_cosine_window()

        # 所有初始化完成后再设 eval 模式
        self.model.eval()

        logger.info(
            f"Tracker 初始化完成 | device={self.device}, "
            f"distractor={'ON' if self.use_distractor else 'OFF'}, "
            f"fusion={'ON' if self.use_fusion else 'OFF'}"
        )

    # ==========================================================
    #  公开接口
    # ==========================================================

    def initialize(self, frame: np.ndarray, bbox: np.ndarray) -> None:
        """
        跟踪初始化（视频首帧调用）。

        步骤:
          1. 裁剪模板区域并缓存模板特征
          2. 初始化干扰物管理器（缓存目标参考特征）
          3. 记录初始目标状态

        参数:
            frame (np.ndarray): 首帧图像, shape = (H, W, 3), BGR
            bbox (np.ndarray):  初始目标框 [x, y, w, h]
                                (x, y) 为左上角坐标, (w, h) 为宽高
        """
        # 转换为 [cx, cy, w, h] 内部格式
        cx = bbox[0] + bbox[2] / 2
        cy = bbox[1] + bbox[3] / 2
        self._target_bbox = np.array([cx, cy, bbox[2], bbox[3]])

        # 裁剪模板区域
        template_patch = self._crop_patch(
            frame, self._target_bbox, self.template_size
        )

        # 模型缓存模板特征
        template_tensor = self._preprocess(template_patch).to(self.device)
        with torch.no_grad():
            self.model.template(template_tensor)

        # 初始化干扰物管理器
        if self.distractor_mgr is not None:
            # 获取目标 CNN 特征（用于干扰物相似度比较的参考）
            target_cnn_feat = self._extract_target_cnn_feature(template_tensor)
            target_img_patch = self._crop_image_patch(frame, bbox)
            self.distractor_mgr.initialize(target_img_patch, target_cnn_feat)

        self._frame_id = 0
        self._is_initialized = True

        logger.info(
            f"跟踪初始化完成 | 目标框: cx={cx:.1f}, cy={cy:.1f}, "
            f"w={bbox[2]:.1f}, h={bbox[3]:.1f}"
        )

    def track(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        单帧跟踪（视频后续帧逐帧调用）。

        参数:
            frame (np.ndarray): 当前帧图像, shape = (H, W, 3), BGR

        返回:
            result (dict):
                "bbox":        目标框 [x, y, w, h] (左上角 + 宽高)
                "score":       跟踪置信度 [0, 1]
                "distractors": 干扰物列表 List[DistractorInfo]
                "adjustment":  融合调整指令 TrackingAdjustment
                "frame_id":    当前帧号
        """
        assert self._is_initialized, "请先调用 initialize() 初始化跟踪器"
        self._frame_id += 1

        # ---- Step 1: 裁剪搜索区域 ----
        search_patch = self._crop_patch(
            frame, self._target_bbox, self.search_size
        )

        # ---- Step 2: 模型前向推理 ----
        search_tensor = self._preprocess(search_patch).to(self.device)
        with torch.no_grad():
            model_output = self.model.track(search_tensor)

        cls_score = model_output["cls_score"]
        bbox_pred = model_output["bbox_pred"]
        attention_map = model_output.get("attention_map", None)

        # ---- Step 3: 解码候选框与置信度 ----
        best_bbox_crop, best_score = self._decode_output(cls_score, bbox_pred)

        # 计算搜索区域几何信息
        search_region_info = self._compute_search_region_info()

        # 将 255x255 上的预测框，映射回原始图像的绝对坐标
        scale = search_region_info["crop_size"] / search_region_info["search_input_size"]
        orig_cx = search_region_info["crop_x1"] + best_bbox_crop[0] * scale
        orig_cy = search_region_info["crop_y1"] + best_bbox_crop[1] * scale
        orig_w = best_bbox_crop[2] * scale
        orig_h = best_bbox_crop[3] * scale

        # 真正在原图上的预测框
        best_bbox_orig = np.array([orig_cx, orig_cy, orig_w, orig_h])

        # ---- Step 4: 干扰物检测 ----
        distractors = []
        if self.distractor_mgr is not None:
            target_xyxy = self._cxcywh_to_xyxy(self._target_bbox)

            distractors = self.distractor_mgr.detect(
                frame=frame,
                target_bbox=target_xyxy,
                attention_map=attention_map,
                search_cnn_features=model_output.get("search_features", None),
                frame_id=self._frame_id,
                search_region_info=search_region_info,
            )

        # ---- Step 5: 融合优化 ----
        adjustment = TrackingAdjustment()
        if self.fusion_module is not None and distractors:
            target_xyxy = self._cxcywh_to_xyxy(self._target_bbox)
            adjustment = self.fusion_module.compute_adjustment(
                target_bbox=target_xyxy,
                target_score=best_score,
                distractors=distractors,
            )

        # ---- Step 6: 应用调整，更新目标状态 (传入原始坐标框) ----
        self._apply_adjustment(best_bbox_orig, best_score, adjustment)

        # 转换为 [x, y, w, h] 输出格式
        cx, cy, w, h = self._target_bbox
        output_bbox = np.array([cx - w / 2, cy - h / 2, w, h])

        result = {
            "bbox": output_bbox,
            "score": best_score,
            "distractors": distractors,
            "adjustment": adjustment,
            "frame_id": self._frame_id,
        }

        logger.debug(
            f"帧 {self._frame_id}: score={best_score:.3f}, "
            f"干扰物={len(distractors)}, "
            f"策略={adjustment.strategy_name}"
        )

        return result

    def reset(self) -> None:
        """重置跟踪器状态（用于切换新视频序列）。"""
        self._target_bbox = None
        self._frame_id = 0
        self._is_initialized = False
        logger.info("跟踪器已重置")

    # ==========================================================
    #  内部方法 — 图像裁剪与预处理
    # ==========================================================

    def _crop_patch(
        self,
        frame: np.ndarray,
        target_bbox: np.ndarray,
        output_size: int,
    ) -> np.ndarray:
        """
        以目标框中心为基准裁剪正方形区域并缩放。

        参数:
            frame:       原始帧图像
            target_bbox: [cx, cy, w, h]
            output_size: 输出正方形边长（127 或 255）

        返回:
            patch: 裁剪并缩放后的图像, shape = (output_size, output_size, 3)
        """
        cx, cy, w, h = target_bbox
        # 计算上下文区域大小（context_amount）
        context = self.context_amount * (w + h)
        base_crop = np.sqrt((w + context) * (h + context))

        # 根据输出尺寸动态调整原图裁剪大小，保持目标相对尺度不变
        crop_size = int(base_crop * (output_size / self.template_size))
        crop_size = max(1, crop_size)  # 防止边界报错

        x1 = int(cx - crop_size / 2)
        y1 = int(cy - crop_size / 2)
        x2 = x1 + crop_size
        y2 = y1 + crop_size

        # 边界填充处理
        H, W = frame.shape[:2]
        # 强制转为 Python 原生 int，防止 OpenCV 底层 C 类型拒绝
        pad_left = int(max(0, -x1))
        pad_top = int(max(0, -y1))
        pad_right = int(max(0, x2 - W))
        pad_bottom = int(max(0, y2 - H))

        # 为了安全，限制最大的 Padding 量（比如不超过图像本身的10倍，防止爆内存）
        max_pad = 5000
        pad_left = min(pad_left, max_pad)
        pad_top = min(pad_top, max_pad)
        pad_right = min(pad_right, max_pad)
        pad_bottom = min(pad_bottom, max_pad)

        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(W, x2)
        y2 = min(H, y2)

        patch = frame[y1:y2, x1:x2]

        if any([pad_left, pad_top, pad_right, pad_bottom]):
            patch = cv2.copyMakeBorder(
                patch, pad_top, pad_bottom, pad_left, pad_right,
                cv2.BORDER_CONSTANT, value=(0, 0, 0)
            )

        # 缩放到目标尺寸
        patch = cv2.resize(patch, (output_size, output_size))
        return patch

    def _crop_image_patch(
        self, frame: np.ndarray, bbox: np.ndarray
    ) -> np.ndarray:
        """裁剪 [x, y, w, h] 格式的图像区域。"""
        x, y, w, h = bbox.astype(int)
        H, W = frame.shape[:2]
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(W, x + w)
        y2 = min(H, y + h)
        return frame[y1:y2, x1:x2]

    def _preprocess(self, patch: np.ndarray) -> torch.Tensor:
        """
        图像预处理: BGR→RGB, HWC→CHW, 归一化到 [0, 1], 增加 batch 维度。

        返回:
            tensor: shape = (1, 3, H, W), dtype = float32
        """
        img = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        # 补充与训练完全一致的 ImageNet 归一化
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img = (img - mean) / std

        img = np.transpose(img, (2, 0, 1))  # HWC → CHW
        tensor = torch.from_numpy(img).unsqueeze(0)  # 增加 batch 维度
        return tensor

    # ==========================================================
    #  内部方法 — 输出解码
    # ==========================================================

    def _decode_output(
        self,
        cls_score: torch.Tensor,
        bbox_pred: torch.Tensor,
    ) -> Tuple[np.ndarray, float]:
        """
        解码 RPN 输出 + 尺度/比例惩罚 → 选出最优目标框。

        SiamRPN++ 标准后处理流程 (Li et al., CVPR 2019):
          1. 解码全部 N 个 anchor → 候选框 [cx, cy, w, h]  (N=1445)
          2. 对每个候选框计算 尺度变化惩罚 + 宽高比变化惩罚
          3. pscore = cls_score × penalty
          4. pscore = pscore × (1 - win_infl) + cosine_window × win_infl
          5. best = argmax(pscore)

           关键: penalty 在 argmax 之前施加到全部候选,
           抑制尺寸剧变的候选, 防止跟踪框宽高爆炸/震荡。

        参数:
            cls_score: shape (1, 2A, H, W)
            bbox_pred: shape (1, 4A, H, W)
        返回:
            best_bbox: [cx,cy,w,h] in 255×255 坐标系
            best_score: 原始(未惩罚)前景概率
        """
        # =========== Step 1: 分类得分 → (N,) ===========
        score = cls_score.squeeze(0)                           # (2A, H, W)
        num_anchors = score.shape[0] // 2
        score = score.view(2, num_anchors, -1).permute(1, 2, 0)  # (A, HW, 2)
        score = torch.softmax(score, dim=-1)[:, :, 1]         # (A, HW)
        score = score.flatten().cpu().numpy()                  # (N,)

        # =========== Step 2: 解码全部 N 个候选框 ===========
        bbox = bbox_pred.squeeze(0)                            # (4A, H, W)
        bbox = bbox.view(4, num_anchors, -1).permute(1, 2, 0) # (A, HW, 4)
        bbox = bbox.reshape(-1, 4).detach().cpu().numpy()      # (N, 4)

        dx, dy, dw, dh = bbox[:, 0], bbox[:, 1], bbox[:, 2], bbox[:, 3]
        dw = np.clip(dw, -2.0, 2.0)    # 防 exp 溢出
        dh = np.clip(dh, -2.0, 2.0)

        anchors = self.anchors                                 # (N, 4) [cx,cy,w,h]
        pred_cx = dx * anchors[:, 2] + anchors[:, 0]
        pred_cy = dy * anchors[:, 3] + anchors[:, 1]
        pred_w  = np.exp(dw) * anchors[:, 2]
        pred_h  = np.exp(dh) * anchors[:, 3]

        # =========== Step 3: 尺度 & 比例惩罚 ===========
        # 目标在 255×255 搜索图中的参考尺寸
        tgt_w, tgt_h = self._target_bbox[2], self._target_bbox[3]
        context = self.context_amount * (tgt_w + tgt_h)
        base_crop = np.sqrt((tgt_w + context) * (tgt_h + context))
        # 补充尺度因子
        crop_sz = max(int(base_crop * (self.search_size / self.template_size)), 1)
        s = self.search_size / crop_sz  # 原图→255 的缩放倍率

        ref_w = tgt_w * s                       # 目标在 255 图中的宽
        ref_h = tgt_h * s                       # 目标在 255 图中的高

        # SiamRPN++ 的 size 函数:  sz(w,h) = sqrt( (w+pad)*(h+pad) ),  pad=(w+h)/2
        def _sz(w, h):
            pad = (w + h) * 0.5
            return np.sqrt((w + pad) * (h + pad))

        # 尺度变化率 (>= 1)
        s_c = np.maximum(
            _sz(pred_w, pred_h) / _sz(ref_w, ref_h),
            _sz(ref_w, ref_h) / _sz(pred_w, pred_h),
        )
        # 宽高比变化率 (>= 1)
        ref_r = ref_w / max(ref_h, 1e-8)
        pred_r = pred_w / np.maximum(pred_h, 1e-8)
        r_c = np.maximum(ref_r / pred_r, pred_r / ref_r)

        # penalty ∈ (0, 1]:  变化越大 → 值越小 → 分数压得越低
        penalty = np.exp(-(r_c * s_c - 1) * self.penalty_k)

        # =========== Step 4: 加惩罚 + 余弦窗 ===========
        pscore = score * penalty

        if self._window is not None and len(self._window) == len(pscore):
            pscore = pscore * (1 - self.window_influence) + \
                     self._window * self.window_influence

        # =========== Step 5: 选最优 ===========
        best_idx = pscore.argmax()
        best_score = float(score[best_idx])   # 返回原始（未惩罚）分类分数

        best_bbox_crop = np.array([
            pred_cx[best_idx], pred_cy[best_idx],
            pred_w[best_idx],  pred_h[best_idx],
        ])

        return best_bbox_crop, best_score

    def _apply_adjustment(
        self,
        raw_bbox: np.ndarray,
        score: float,
        adjustment: TrackingAdjustment,
    ) -> None:
        """
        将原始跟踪结果 + 融合调整 → 更新目标状态。

        参数:
            raw_bbox:   RPN 解码的原始目标框 [cx, cy, w, h]
            score:      跟踪置信度
            adjustment: 融合模块的调整指令
        """
        # 基础更新（平滑尺度变化）
        cx, cy = raw_bbox[0], raw_bbox[1]
        w = self._target_bbox[2] * (1 - self.scale_lr) + raw_bbox[2] * self.scale_lr
        h = self._target_bbox[3] * (1 - self.scale_lr) + raw_bbox[3] * self.scale_lr

        # 应用融合偏移
        cx += adjustment.bbox_offset[0]
        cy += adjustment.bbox_offset[1]

        # 设置尺寸的下限和上限，防止跟踪框过小或过大导致后续处理异常
        # 最小值限制为 10 像素，最大值限制为 4000 像素 (防止 Infinity 溢出)
        w = min(max(10.0, float(w)), 4000.0)
        h = min(max(10.0, float(h)), 4000.0)

        self._target_bbox = np.array([cx, cy, w, h])

    # ==========================================================
    #  工具方法
    # ==========================================================

    def _create_cosine_window(self) -> Optional[np.ndarray]:
        """创建余弦窗，用于跟踪框位置平滑。"""
        # SiamRPN++ 分辨率固定为 17x17
        size = 17
        hanning = np.outer(np.hanning(size), np.hanning(size))
        window = np.tile(hanning.flatten(), 5)
        # 窗口最大值归一化为 1.0
        return window / window.max()

    def _extract_target_cnn_feature(
        self, template_tensor: torch.Tensor
    ) -> torch.Tensor:
        """
        提取目标区域的 CNN 特征向量（供干扰物管理器作为参考基准）。

        模板图像本身就是以目标为中心的紧密裁剪，因此对 layer3 特征图
        做全局平均池化即可得到目标的全局语义特征向量。
        该向量将与 roi_align 提取的候选区域特征做余弦相似度比较。

        返回:
            feat: shape = (C_layer3,) 的 1D 特征向量（如 ResNet50 → 1024 维）
        """
        with torch.no_grad():
            features = self.model.backbone(template_tensor)
            layer_name = self.model.backbone.output_layers[1]  # layer3
            feat = features[layer_name]
            # 全局平均池化 → 1D 向量
            feat = torch.nn.functional.adaptive_avg_pool2d(feat, 1).flatten()
        return feat

    def _compute_search_region_info(self) -> dict:
        """
        计算当前帧搜索区域裁剪的几何信息。

           crop_size / crop_x1 / crop_y1 必须用 int() 截断，
           与 _crop_patch 实际裁剪的整数边长完全一致，
           否则 255→原图 坐标映射会产生偏差并逐帧累积。
        """
        cx, cy, w, h = self._target_bbox

        context = self.context_amount * (w + h)
        base_crop = np.sqrt((w + context) * (h + context))
        # 补充尺度因子
        crop_size = int(base_crop * (self.search_size / self.template_size))
        crop_size = max(1, crop_size)

        crop_x1 = int(cx - crop_size / 2)                         
        crop_y1 = int(cy - crop_size / 2)                    

        return {
            "crop_x1": float(crop_x1),
            "crop_y1": float(crop_y1),
            "crop_size": float(crop_size),
            "search_input_size": self.search_size,
            "feature_stride": self.feature_stride,
        }

    @staticmethod
    def _cxcywh_to_xyxy(bbox: np.ndarray) -> np.ndarray:
        """[cx, cy, w, h] → [x1, y1, x2, y2]"""
        cx, cy, w, h = bbox
        return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])

    @staticmethod
    def load_config(config_path: str) -> dict:
        """加载 YAML 配置文件。"""
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        logger.info(f"配置已加载: {config_path}")
        return cfg