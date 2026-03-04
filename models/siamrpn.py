"""
siamrpn.py — SiamRPN++ 完整模型封装
======================================
将 Backbone + CBAM + Neck + RPN 组装为端到端模型。

架构流程:
  ┌────────────┐    ┌────────────┐
  │ 模板图像 z │    │ 搜索图像 x │
  └─────┬──────┘    └─────┬──────┘
        │                  │
        ▼                  ▼
  ┌─────────────── Backbone (共享权重) ──────────────┐
  │  layer2, layer3, layer4  │  layer2, layer3, layer4│
  └─────┬───────────────────────────┬────────────────┘
        │                           │
        │                    ┌──────▼──────┐
        │                    │    CBAM     │ ← 仅搜索分支
        │                    │ (layer3后)  │
        │                    └──────┬──────┘
        │                           │ + attention_map (供干扰物挖掘)
        ▼                           ▼
  ┌─── Neck (通道调整) ───┐  ┌─── Neck (通道调整) ───┐
  │   256ch × 3 layers    │  │   256ch × 3 layers    │
  └─────┬─────────────────┘  └─────┬─────────────────┘
        │                           │
        └───────────┬───────────────┘
                    ▼
           ┌──── MultiRPN ────┐
           │ 深度互相关 + 融合 │
           └────┬────────┬────┘
                │        │
                ▼        ▼
           cls_score  bbox_pred
"""

import logging
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from .backbone import build_backbone
from .cbam import CBAM
from .neck import AdjustAllLayer
from .rpn import MultiRPN

logger = logging.getLogger(__name__)


class SiamRPNPP(nn.Module):
    """
    SiamRPN++ 完整模型（含 CBAM 注意力增强）。

    参数:
        cfg (dict): 完整配置字典（包含 backbone, cbam, rpn 等节）
    """

    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = cfg

        # ---- 1. 骨干网络（模板 & 搜索分支共享） ----
        self.backbone = build_backbone(cfg)
        backbone_out_channels = self.backbone.get_out_channels()

        # ---- 2. CBAM 注意力模块（仅用于搜索分支） ----
        cbam_cfg = cfg.get("cbam", {})
        self.use_cbam = cbam_cfg.get("enabled", True)
        if self.use_cbam:
            # CBAM 作用于 layer3 输出（默认通道数 = backbone_out_channels[1]）
            cbam_channels = backbone_out_channels[1]  # layer3 的通道数
            self.cbam = CBAM(
                in_channels=cbam_channels,
                reduction_ratio=cbam_cfg.get("reduction_ratio", 16),
                spatial_kernel_size=cbam_cfg.get("spatial_kernel_size", 7),
            )
            logger.info(f"CBAM 已启用 | 作用通道数: {cbam_channels}")
        else:
            self.cbam = None
            logger.info("CBAM 已禁用")

        # ---- 3. Neck 特征调整层（模板 & 搜索各一套） ----
        rpn_in_channels = cfg["rpn"]["in_channels"]
        self.neck = AdjustAllLayer(backbone_out_channels, rpn_in_channels)

        # ---- 4. 多尺度 RPN 头 ----
        num_anchors = len(cfg["rpn"]["anchor_ratios"]) * len(cfg["rpn"]["anchor_scales"])
        self.rpn = MultiRPN(
            in_channels=rpn_in_channels,
            cls_out_channels=cfg["rpn"]["cls_out_channels"],
            num_anchors=num_anchors,
            num_layers=len(backbone_out_channels),
        )

        logger.info("SiamRPN++ 模型构建完成")

    # ==========================================================
    #  模板分支：初始化跟踪时调用一次
    # ==========================================================
    def template(self, z: torch.Tensor) -> None:
        """
        提取并缓存模板特征（跟踪初始化时调用一次）。

        参数:
            z: 模板图像裁剪, shape = (1, 3, 127, 127)
        """
        # 骨干网络提取多尺度特征
        z_features = self.backbone(z)
        z_feat_list = [z_features[name] for name in self.backbone.output_layers]

        # Neck 调整通道数
        self.z_feats = self.neck(z_feat_list)

        logger.debug(f"模板特征已缓存 | 特征层数: {len(self.z_feats)}")

    # ==========================================================
    #  搜索分支：每帧调用
    # ==========================================================
    def track(
        self, x: torch.Tensor
    ) -> Dict[str, Any]:
        """
        对搜索区域执行特征提取 + CBAM + 互相关，输出跟踪结果。

        参数:
            x: 搜索区域裁剪, shape = (1, 3, 255, 255)

        返回:
            result (dict):
                "cls_score": 分类得分图, shape = (1, 2*A, H, W)
                "bbox_pred": 回归预测图, shape = (1, 4*A, H, W)
                "attention_map": CBAM 空间注意力图 (可选), shape = (1, 1, H3, W3)
                    用于干扰物挖掘模块定位低关注度区域
        """
        result = {}

        # 骨干网络提取搜索区域多尺度特征
        x_features = self.backbone(x)
        x_feat_list = [x_features[name] for name in self.backbone.output_layers]

        #   缓存 CBAM 处理前的 layer3 原始特征图，供干扰物挖掘模块使用
        #   shape = (1, C_layer3, H_feat, W_feat)
        #   该特征图与 255×255 搜索输入的空间对应关系为 stride=16
        result["search_features"] = x_feat_list[1].clone()

        # CBAM 注意力增强（作用于 layer3，即列表的第 [1] 个元素）
        attention_map = None
        if self.use_cbam and self.cbam is not None:
            x_feat_list[1], attention_map = self.cbam(x_feat_list[1])
            result["attention_map"] = attention_map

        # Neck 调整通道数
        x_feats = self.neck(x_feat_list)

        # MultiRPN 深度互相关 + 融合
        cls_score, bbox_pred = self.rpn(self.z_feats, x_feats)
        result["cls_score"] = cls_score
        result["bbox_pred"] = bbox_pred

        return result

    # ==========================================================
    #  训练前向（同时接收模板和搜索图像）
    # ==========================================================
    def forward(
        self,
        template: torch.Tensor,
        search: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        训练模式的前向传播。

        参数:
            template: 模板图像, shape = (B, 3, 127, 127)
            search:   搜索图像, shape = (B, 3, 255, 255)

        返回:
            outputs (dict): 包含 cls_score, bbox_pred, attention_map
        """
        # 模板分支
        z_features = self.backbone(template)
        z_feat_list = [z_features[name] for name in self.backbone.output_layers]
        z_feats = self.neck(z_feat_list)

        # 搜索分支
        x_features = self.backbone(search)
        x_feat_list = [x_features[name] for name in self.backbone.output_layers]

        outputs = {}
        if self.use_cbam and self.cbam is not None:
            x_feat_list[1], attention_map = self.cbam(x_feat_list[1])
            outputs["attention_map"] = attention_map

        x_feats = self.neck(x_feat_list)

        # RPN
        cls_score, bbox_pred = self.rpn(z_feats, x_feats)
        outputs["cls_score"] = cls_score
        outputs["bbox_pred"] = bbox_pred

        return outputs