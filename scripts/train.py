"""
train.py — 训练入口脚本
=========================
训练 SiamRPN++ + CBAM + 干扰物加权损失 模型。

用法:
    python scripts/train.py --config configs/default.yaml
    python scripts/train.py --config configs/default.yaml --resume checkpoints/latest.pth

训练数据流:
    序列 → 帧对采样 → 裁剪增强 → Anchor 标签生成 + 干扰物掩码
         → 模型前向 → cls_loss + reg_loss + distractor_loss → 反向传播

保存策略:
    latest.pth  — 每个 Epoch 结束保存 (覆盖)
    epoch_N.pth — 每 10 个 Epoch 额外保存
    best.pth    — 基于 GOT-10k 验证集指标保存最优模型
                  (每 Epoch 计算 val_loss, 每 val_auc_interval Epoch 计算 AUC)
"""

import argparse
import logging
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from typing import Optional
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.siamrpn import SiamRPNPP
from core.tracker import Tracker
from core.fusion import FusionModule
from data.datasets import build_dataset, GOT10kDataset, BaseTrackingDataset
from data.train_dataset import TrackingTrainDataset
from utils.logger import setup_logger
from utils.metrics import Evaluator

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="SiamRPN++ 干扰物感知跟踪 — 模型训练")
    parser.add_argument("--config", type=str, default="configs/default.yaml",
                        help="配置文件路径")
    parser.add_argument("--resume", type=str, default=None,
                        help="恢复训练的检查点路径")
    return parser.parse_args()


def build_train_loader(cfg: dict) -> DataLoader:
    """
    构建训练 DataLoader。

    根据配置中 train.datasets 列表加载一个或多个数据集，
    合并所有序列后构建 TrackingTrainDataset。
    """
    train_cfg = cfg["train"]
    dataset_names = train_cfg.get("datasets", ["otb100"])

    # 加载数据集实例
    dataset_instances = []
    for name in dataset_names:
        try:
            ds = build_dataset(cfg, name)
            if len(ds) > 0:
                dataset_instances.append(ds)
                logger.info(f"训练数据集 [{name}] 加载成功: {len(ds)} 个序列")
            else:
                logger.warning(f"训练数据集 [{name}] 为空, 跳过")
        except (ValueError, FileNotFoundError) as e:
            logger.warning(f"训练数据集 [{name}] 加载失败: {e}")

    if not dataset_instances:
        raise RuntimeError(
            "没有可用的训练数据集! 请检查 configs/default.yaml 中 "
            "datasets 和 train.datasets 的路径配置。"
        )

    # 构建训练数据集
    train_dataset = TrackingTrainDataset.from_config(cfg, dataset_instances)

    # 构建 DataLoader
    loader = DataLoader(
        train_dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        num_workers=cfg["system"].get("num_workers", 4),
        pin_memory=True,
        drop_last=True,
    )

    logger.info(
        f"DataLoader 构建完成 | "
        f"样本数={len(train_dataset)}, "
        f"batch_size={train_cfg['batch_size']}, "
        f"batches/epoch={len(loader)}"
    )

    return loader


# ==============================================================
#  构建 GOT-10k 验证集 DataLoader
# ==============================================================

def build_val_loader(cfg: dict) -> Optional[DataLoader]:
    """
    构建 GOT-10k 验证集 DataLoader。

    GOT-10k 标准目录结构:
        data/GOT-10k/
        ├── train/    (~9,335 序列, 用于训练)
        └── val/      (~180 序列, 用于验证)

    如果 val/ 目录不存在或为空, 返回 None (优雅降级)。
    """
    train_cfg = cfg["train"]
    datasets_cfg = cfg.get("datasets", {})
    got10k_cfg = datasets_cfg.get("got10k", {})
    got10k_root = got10k_cfg.get("root", "data/GOT-10k/")

    # GOT-10k 验证集路径
    val_root = os.path.join(got10k_root, "val")
    if not os.path.isdir(val_root):
        logger.warning(
            f"GOT-10k 验证集目录不存在: {val_root} | "
            f"跳过验证，best.pth 将基于训练损失保存。"
            f"建议下载 GOT-10k val 集: http://got-10k.aitestunion.com/downloads"
        )
        return None

    # 加载验证集序列
    try:
        val_dataset_raw = GOT10kDataset(val_root)
        if len(val_dataset_raw) == 0:
            logger.warning("GOT-10k 验证集为空, 跳过验证")
            return None
    except Exception as e:
        logger.warning(f"GOT-10k 验证集加载失败: {e}")
        return None

    # 构建 TrackingTrainDataset (复用训练数据流水线, 但不做增强的帧对采样)
    val_train_dataset = TrackingTrainDataset(
        sequences=val_dataset_raw.sequences,
        cfg=cfg,
    )

    # 验证集用较小的 batch_size, 不 drop_last
    val_batch_size = min(train_cfg["batch_size"], 16)
    val_loader = DataLoader(
        val_train_dataset,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=min(cfg["system"].get("num_workers", 4), 2),
        pin_memory=True,
        drop_last=False,
    )

    logger.info(
        f"验证 DataLoader 构建完成 | "
        f"GOT-10k val 序列数={len(val_dataset_raw)}, "
        f"帧对数={len(val_train_dataset)}, "
        f"batches={len(val_loader)}"
    )

    return val_loader


# ==============================================================
#  验证函数 (计算 val_loss)
# ==============================================================

@torch.no_grad()
def validate(
    model: nn.Module,
    val_loader: DataLoader,
    num_anchors: int,
    fusion_module: FusionModule,
    cfg: dict,
    device: str,
    max_batches: int = 50,
) -> dict:
    """
    在 GOT-10k 验证集上计算损失 (不更新梯度)。

    为控制验证时间, 默认最多跑 max_batches 个 batch。
    GOT-10k val 约 180 序列, 足够获得稳定的验证信号。

    参数:
        model:         SiamRPN++ 模型
        val_loader:    验证 DataLoader
        num_anchors:   每个位置的 anchor 数量
        fusion_module: FusionModule 实例
        cfg:           配置字典
        device:        设备
        max_batches:   最大验证 batch 数 (控制验证时间)

    返回:
        val_metrics (dict):
            "val_loss":     加权总损失
            "val_cls_loss": 分类损失
            "val_reg_loss": 回归损失
            "val_dist_loss": 干扰物加权损失
    """
    model.eval()

    total_losses = {"cls": 0.0, "reg": 0.0, "dist": 0.0, "total": 0.0}
    num_batches = 0

    for batch_idx, batch in enumerate(val_loader):
        if batch_idx >= max_batches:
            break

        template = batch["template"].to(device)
        search = batch["search"].to(device)
        cls_label = batch["cls_label"].to(device)
        reg_label = batch["reg_label"].to(device)
        distractor_mask = batch["distractor_mask"].to(device)

        # 前向传播
        outputs = model(template, search)
        cls_score = outputs["cls_score"]
        bbox_pred = outputs["bbox_pred"]

        # 重塑
        cls_score, bbox_pred = reshape_rpn_output(cls_score, bbox_pred, num_anchors)

        # 计算损失
        losses = compute_losses(
            cls_score, bbox_pred,
            cls_label, reg_label, distractor_mask,
            fusion_module, cfg,
        )

        total_losses["cls"] += losses["cls_loss"].item()
        total_losses["reg"] += losses["reg_loss"].item()
        total_losses["dist"] += losses["dist_loss"].item()
        total_losses["total"] += losses["total_loss"].item()
        num_batches += 1

    n = max(num_batches, 1)
    val_metrics = {
        "val_loss": total_losses["total"] / n,
        "val_cls_loss": total_losses["cls"] / n,
        "val_reg_loss": total_losses["reg"] / n,
        "val_dist_loss": total_losses["dist"] / n,
    }

    return val_metrics


# ==============================================================
#  AUC 评估函数 (完整跟踪推理)
# ==============================================================

@torch.no_grad()
def evaluate_auc(
    model: nn.Module,
    cfg: dict,
    max_sequences: int = 20,
) -> Optional[float]:
    """
    在 GOT-10k 验证集上运行完整跟踪推理, 计算 AUC。

    由于完整跟踪推理较慢, 默认只评估前 max_sequences 个序列。
    GOT-10k val 的 180 个序列大多较短 (几十~几百帧), 20 个序列
    约需 1-3 分钟, 足够反映模型跟踪能力。

    参数:
        model:          SiamRPN++ 模型
        cfg:            配置字典
        max_sequences:  最大评估序列数 (控制评估时间)

    返回:
        mean_auc:  平均 AUC 值, 或 None (验证集不可用时)
    """
    datasets_cfg = cfg.get("datasets", {})
    got10k_cfg = datasets_cfg.get("got10k", {})
    got10k_root = got10k_cfg.get("root", "data/GOT-10k/")
    val_root = os.path.join(got10k_root, "val")

    if not os.path.isdir(val_root):
        return None

    try:
        val_dataset = GOT10kDataset(val_root)
        if len(val_dataset) == 0:
            return None
    except Exception:
        return None

    # 构建 Tracker (使用当前模型权重)
    tracker = Tracker(cfg)
    # 将训练中的模型权重复制到 tracker 的模型
    tracker.model.load_state_dict(model.state_dict())
    tracker.model.eval()

    evaluator = Evaluator(cfg)

    # 限制评估序列数
    sequences_to_eval = val_dataset.sequences[:max_sequences]

    for seq in sequences_to_eval:
        if len(seq) < 2:
            continue

        evaluator.start_sequence(seq.name)

        # 加载首帧并初始化
        first_frame = val_dataset.load_frame(seq.frame_paths[0])
        init_bbox = seq.ground_truth[0]  # [x, y, w, h]

        if init_bbox[2] <= 0 or init_bbox[3] <= 0:
            evaluator.end_sequence()
            tracker.reset()
            continue

        tracker.initialize(first_frame, init_bbox)

        # 逐帧跟踪
        for i in range(1, len(seq)):
            frame = val_dataset.load_frame(seq.frame_paths[i])
            gt_bbox = seq.ground_truth[i]

            evaluator.start_timer()

            result = tracker.track(frame)

            elapsed = evaluator.stop_timer()
            evaluator.update(result["bbox"], gt_bbox, elapsed)

        evaluator.end_sequence()
        tracker.reset()

    report = evaluator.compute_report()
    if not report:
        return None

    mean_auc = report["overall"]["mean_auc"]
    return mean_auc


def reshape_rpn_output(
    cls_score: torch.Tensor,
    bbox_pred: torch.Tensor,
    num_anchors: int,
) -> tuple:
    """
    将 RPN 输出从 (B, C*A, H, W) 格式重塑为训练所需格式。

    RPN 原始输出:
        cls_score: (B, 2*A, H, W)   例 (16, 10, 17, 17)
        bbox_pred: (B, 4*A, H, W)   例 (16, 20, 17, 17)

    重塑后（与 anchor 标签排列一致）:
        cls_score: (B, 2, A*H*W)    例 (16, 2, 1445)
        bbox_pred: (B, 4, A*H*W)    例 (16, 4, 1445)

    排列顺序: anchor_0@(0,0), anchor_0@(0,1), ..., anchor_{A-1}@(H-1,W-1)
    """
    B = cls_score.size(0)
    H, W = cls_score.shape[2], cls_score.shape[3]

    # (B, 2*A, H, W) → (B, 2, A, H, W) → (B, 2, A*H*W)
    cls_score = cls_score.view(B, 2, num_anchors, H, W)
    cls_score = cls_score.reshape(B, 2, -1)

    # (B, 4*A, H, W) → (B, 4, A, H, W) → (B, 4, A*H*W)
    bbox_pred = bbox_pred.view(B, 4, num_anchors, H, W)
    bbox_pred = bbox_pred.reshape(B, 4, -1)

    return cls_score, bbox_pred


def compute_losses(
    cls_score: torch.Tensor,
    bbox_pred: torch.Tensor,
    cls_label: torch.Tensor,
    reg_label: torch.Tensor,
    distractor_mask: torch.Tensor,
    fusion_module: FusionModule,
    cfg: dict,
) -> dict:
    """
    计算三项损失: 分类 + 回归 + 干扰物加权。

    将三项损失彻底分离:
      - cls_loss:  标准分类 CE, 作用于非干扰物的 valid anchor
      - dist_loss: 困难负样本 CE, 仅作用于干扰物 anchor (IoU [0.3,0.6))
      - reg_loss:  正样本回归 SmoothL1 (不变)

    这样 cls_loss 负责基本前景/背景分类,
    dist_loss 专注困难干扰物区分, 两者不再重叠。
    """
    train_cfg = cfg["train"]
    cls_w = train_cfg.get("cls_loss_weight", 1.0)
    reg_w = train_cfg.get("reg_loss_weight", 1.2)
    dist_w = train_cfg.get("dist_loss_weight", 0.5)

    B, _, N = cls_score.shape
    device = cls_score.device
    label_smoothing = train_cfg.get("label_smoothing", 0.0)

    # ============ 分类损失 (排除干扰物 anchor) ============
    # 干扰物 anchor 交给 dist_loss 单独处理, 避免重复计算
    cls_label_for_ce = cls_label.clone()
    cls_label_for_ce[distractor_mask > 0] = -1  # 在 cls_loss 中 ignore

    cls_loss = nn.functional.cross_entropy(
        cls_score, cls_label_for_ce.long(), ignore_index=-1,
        reduction="mean", label_smoothing=label_smoothing
    )

    # ============ 回归损失 (仅正样本, 不变) ============
    pos_mask = cls_label == 1
    num_pos = max(pos_mask.sum().item(), 1)

    pos_mask_4d = pos_mask.unsqueeze(1).expand_as(bbox_pred)
    reg_loss = nn.functional.smooth_l1_loss(
        bbox_pred[pos_mask_4d],
        reg_label[pos_mask_4d],
        reduction="sum",
    ) / num_pos

    # ============ 干扰物损失 (仅困难负样本) ============
    dist_anchor_mask = (distractor_mask > 0) & (cls_label >= 0)
    num_dist = max(dist_anchor_mask.sum().item(), 1)

    if dist_anchor_mask.any():
        per_element_ce = nn.functional.cross_entropy(
            cls_score, cls_label.clamp(min=0).long(), reduction="none"
        )
        dist_loss = (
            per_element_ce * dist_anchor_mask.float()
        ).sum() / num_dist * fusion_module.distractor_loss_weight
    else:
        dist_loss = torch.tensor(0.0, device=device)

    # ============ 加权总损失 ============
    total_loss = cls_w * cls_loss + reg_w * reg_loss + dist_w * dist_loss

    return {
        "cls_loss": cls_loss,
        "reg_loss": reg_loss,
        "dist_loss": dist_loss,
        "total_loss": total_loss,
    }


def main():
    args = parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    setup_logger(cfg)
    logger.info("=" * 60)
    logger.info("  SiamRPN++ 干扰物感知跟踪 — 开始训练")
    logger.info("=" * 60)

    device = cfg["system"]["device"]
    train_cfg = cfg["train"]

    # ---- 构建模型 ----
    model = SiamRPNPP(cfg).to(device)
    model.train()

    total_params = sum(p.numel() for p in model.parameters())
    train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"模型参数量: 总计 {total_params:,} | 可训练 {train_params:,}")

    # ---- 从配置推导 anchor 数量 ----
    num_anchors = len(cfg["rpn"]["anchor_scales"]) * len(cfg["rpn"]["anchor_ratios"])

    # ---- 优化器 (分层学习率) ----
    #   Backbone 使用更低的学习率, 保护预训练特征不被洗掉
    #   若所有可训练参数统一 lr=0.001 → 预训练权重崩塌
    #   backbone_lr = lr × backbone_lr_mult (默认 0.1)
    backbone_lr_mult = train_cfg.get("backbone_lr_mult", 0.1)
    base_lr = train_cfg["lr"]

    backbone_params = []
    head_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("backbone."):
            backbone_params.append(param)
        else:
            head_params.append(param)

    param_groups = [
        {"params": backbone_params, "lr": base_lr * backbone_lr_mult, "name": "backbone"},
        {"params": head_params,     "lr": base_lr,                    "name": "head"},
    ]

    optimizer = optim.AdamW(
        param_groups,
        lr=base_lr,  # default lr (head_params 会用这个)
        weight_decay=train_cfg["weight_decay"],
    )

    logger.info(
        f"分层学习率: backbone_lr={base_lr * backbone_lr_mult:.6f} "
        f"({len(backbone_params)} params), "
        f"head_lr={base_lr:.6f} ({len(head_params)} params)"
    )

    # ---- 学习率调度器 ----
    if train_cfg["lr_scheduler"] == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=train_cfg["epochs"]
        )
    else:
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)

    # ---- 融合模块（提供 distractor loss 计算）----
    fusion_module = FusionModule(cfg)

    # ---- 恢复训练 ----
    start_epoch = 0
    best_metric = float("inf")     # 最优指标 (val_loss 越低越好)
    best_auc = 0.0                 # 最优 AUC (越高越好)
    best_metric_type = "val_loss"  # 当前使用的最优指标类型
    # early stopping 相关
    early_stop_patience = train_cfg.get("early_stop_patience", 10)
    no_improve_count = 0
    if args.resume and os.path.exists(args.resume):
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_epoch = checkpoint.get("epoch", 0)
        best_metric = checkpoint.get("best_metric", float("inf"))
        best_auc = checkpoint.get("best_auc", 0.0)
        best_metric_type = checkpoint.get("best_metric_type", "val_loss")
        no_improve_count = checkpoint.get("no_improve_count", 0)
        best_val_str = f"{best_auc:.4f}" if best_metric_type == "auc" else f"{best_metric:.4f}"
        logger.info(
            f"从 epoch {start_epoch} 恢复训练: {args.resume} | "
            f"best_{best_metric_type}={best_val_str}"
        )

    # ---- 构建训练 DataLoader ----
    train_loader = build_train_loader(cfg)

    # ---- 构建验证 DataLoader (GOT-10k val) ----
    val_loader = build_val_loader(cfg)

    # ---- 验证参数 ----
    val_auc_interval = train_cfg.get("val_auc_interval", 5)  # 每 N 个 epoch 做 AUC 评估
    val_auc_max_seqs = train_cfg.get("val_auc_max_seqs", 20)  # AUC 评估的最大序列数
    val_max_batches = train_cfg.get("val_max_batches", 50)     # val_loss 评估的最大 batch 数

    # ---- Warmup 调度器 ----
    warmup_epochs = train_cfg.get("warmup_epochs", 5)
    # base_lr 和 backbone_lr_mult 已在优化器构建时定义

    logger.info(
        f"训练配置: epochs={train_cfg['epochs']}, "
        f"batch_size={train_cfg['batch_size']}, "
        f"lr={train_cfg['lr']}, "
        f"warmup={warmup_epochs} epochs"
    )
    logger.info(
        f"验证配置: val_loader={'已构建' if val_loader else '不可用'}, "
        f"auc_interval={val_auc_interval} epochs, "
        f"auc_max_seqs={val_auc_max_seqs}"
    )

    # ================================================================
    #  训练循环
    # ================================================================

    ckpt_dir = cfg["system"]["checkpoint_dir"]
    os.makedirs(ckpt_dir, exist_ok=True)
    best_path = os.path.join(ckpt_dir, "best.pth")

    for epoch in range(start_epoch, train_cfg["epochs"]):
        model.train()

        # 每个 epoch 重新采样帧对
        train_loader.dataset.reshuffle()

        epoch_losses = {"cls": 0.0, "reg": 0.0, "dist": 0.0, "total": 0.0}
        num_batches = 0
        t_epoch_start = time.time()

        for batch_idx, batch in enumerate(train_loader):
            t_batch_start = time.time()

            # ---- 数据搬到 GPU ----
            template = batch["template"].to(device)           # (B, 3, 127, 127)
            search = batch["search"].to(device)               # (B, 3, 255, 255)
            cls_label = batch["cls_label"].to(device)         # (B, N)
            reg_label = batch["reg_label"].to(device)         # (B, 4, N)
            distractor_mask = batch["distractor_mask"].to(device)  # (B, N)

            # ---- Warmup: 前几个 epoch 线性增加学习率 ----
            if epoch < warmup_epochs:
                warmup_progress = (
                    (epoch * len(train_loader) + batch_idx)
                    / (warmup_epochs * len(train_loader))
                )
                for pg in optimizer.param_groups:
                    # 分层 warmup: 每个 param group 按自身基准 LR 线性增长
                    if pg.get("name") == "backbone":
                        pg["lr"] = base_lr * backbone_lr_mult * warmup_progress
                    else:
                        pg["lr"] = base_lr * warmup_progress

            # ---- 前向传播 ----
            outputs = model(template, search)
            cls_score = outputs["cls_score"]   # (B, 2*A, H, W)
            bbox_pred = outputs["bbox_pred"]   # (B, 4*A, H, W)

            # ---- 重塑 RPN 输出，对齐 anchor 排列 ----
            cls_score, bbox_pred = reshape_rpn_output(
                cls_score, bbox_pred, num_anchors
            )
            # cls_score: (B, 2, A*H*W), bbox_pred: (B, 4, A*H*W)

            # ---- 计算损失 ----
            losses = compute_losses(
                cls_score, bbox_pred,
                cls_label, reg_label, distractor_mask,
                fusion_module, cfg,
            )

            total_loss = losses["total_loss"]

            # ---- 反向传播 ----
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), train_cfg["grad_clip"]
            )
            optimizer.step()

            # ---- 统计 ----
            epoch_losses["cls"] += losses["cls_loss"].item()
            epoch_losses["reg"] += losses["reg_loss"].item()
            epoch_losses["dist"] += losses["dist_loss"].item()
            epoch_losses["total"] += total_loss.item()
            num_batches += 1

            # ---- 批次日志 (每 10 批打印一次) ----
            if (batch_idx + 1) % 10 == 0:
                batch_time = time.time() - t_batch_start
                current_lr = optimizer.param_groups[-1]["lr"]  # head LR
                logger.info(
                    f"  Epoch [{epoch+1}/{train_cfg['epochs']}] "
                    f"Batch [{batch_idx+1}/{len(train_loader)}] | "
                    f"Loss: {total_loss.item():.4f} "
                    f"(cls={losses['cls_loss'].item():.3f}, "
                    f"reg={losses['reg_loss'].item():.3f}, "
                    f"dist={losses['dist_loss'].item():.3f}) | "
                    f"LR: {current_lr:.6f} | "
                    f"Time: {batch_time:.2f}s"
                )

        # ---- Epoch 结束: 更新学习率 ----
        if epoch >= warmup_epochs:
            scheduler.step()

        # ---- 训练损失统计 ----
        train_time = time.time() - t_epoch_start
        n = max(num_batches, 1)
        avg_train_loss = epoch_losses["total"] / n

        # ============================================================
        #  验证阶段
        # ============================================================

        val_loss = None
        val_auc = None

        # --- 1. 每个 Epoch 计算 val_loss (快速, ~30秒) ---
        if val_loader is not None:
            t_val_start = time.time()
            val_metrics = validate(
                model, val_loader, num_anchors,
                fusion_module, cfg, device,
                max_batches=val_max_batches,
            )
            val_loss = val_metrics["val_loss"]
            val_time = time.time() - t_val_start

            logger.info(
                f"  验证 [val_loss] | "
                f"Loss: {val_loss:.4f} "
                f"(cls={val_metrics['val_cls_loss']:.3f}, "
                f"reg={val_metrics['val_reg_loss']:.3f}, "
                f"dist={val_metrics['val_dist_loss']:.3f}) | "
                f"Time: {val_time:.1f}s"
            )

        # --- 2. 每 val_auc_interval Epoch 计算 AUC (较慢, 1~3分钟) ---
        if (epoch + 1) % val_auc_interval == 0:
            t_auc_start = time.time()
            logger.info(
                f"  开始 AUC 评估 (GOT-10k val, 最多 {val_auc_max_seqs} 序列)..."
            )
            auc_result = evaluate_auc(model, cfg, max_sequences=val_auc_max_seqs)
            auc_time = time.time() - t_auc_start

            if auc_result is not None:
                val_auc = auc_result
                logger.info(
                    f"  验证 [AUC] | "
                    f"mean_AUC: {val_auc:.4f} | "
                    f"Time: {auc_time:.1f}s"
                )

            # AUC 评估后恢复训练模式
            model.train()

        # ============================================================
        #  best.pth 保存判断
        # ============================================================

        is_best = False
        current_metric_str = ""

        # 优先使用 AUC (如果本轮有计算)
        if val_auc is not None:
            if val_auc > best_auc:
                best_auc = val_auc
                best_metric_type = "auc"
                is_best = True
            current_metric_str = f"AUC={val_auc:.4f} (best={best_auc:.4f})"

        # 否则使用 val_loss
        elif val_loss is not None:
            if val_loss < best_metric:
                best_metric = val_loss
                # 仅在从未有过 AUC 评估时才基于 val_loss 判定 best
                if best_metric_type == "val_loss":
                    is_best = True
            current_metric_str = f"val_loss={val_loss:.4f} (best={best_metric:.4f})"

        # 都不可用时, 使用训练损失 (降级方案)
        else:
            if avg_train_loss < best_metric:
                best_metric = avg_train_loss
                is_best = True
            current_metric_str = f"train_loss={avg_train_loss:.4f} (best={best_metric:.4f})"

        # ---- 构建通用 checkpoint 字典 ----
        ckpt_dict = {
            "epoch": epoch + 1,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "best_metric": best_metric,
            "best_auc": best_auc,
            "best_metric_type": best_metric_type,
            "no_improve_count": no_improve_count,
        }

        # ---- 保存 latest.pth (每个 Epoch) ----
        latest_path = os.path.join(ckpt_dir, "latest.pth")
        torch.save(ckpt_dict, latest_path)

        # ---- 保存 epoch_N.pth (每 10 Epoch) ----
        if (epoch + 1) % 10 == 0:
            torch.save(ckpt_dict, os.path.join(ckpt_dir, f"epoch_{epoch+1}.pth"))

        # ---- 保存 best.pth (验证指标改善时) ----
        if is_best:
            torch.save(ckpt_dict, best_path)
            logger.info(f"  ★ 保存最优模型 → {best_path} ({current_metric_str})")

            no_improve_count = 0
        else:
            no_improve_count += 1

        # early stopping check
        if no_improve_count >= early_stop_patience and epoch >= train_cfg.get("warmup_epochs", 5) + early_stop_patience:
            logger.info(
                f"  Early stopping: no improvement for {no_improve_count} epochs, "
                f"stopping at epoch {epoch+1}"
            )
            break

        # ---- Epoch 日志 ----
        lr_info = " | ".join(
            f"{pg.get('name', 'default')}_lr={pg['lr']:.6f}"
            for pg in optimizer.param_groups
        )
        logger.info(
            f"Epoch [{epoch+1}/{train_cfg['epochs']}] 完成 | "
            f"train_loss: {avg_train_loss:.4f} "
            f"(cls={epoch_losses['cls']/n:.3f}, "
            f"reg={epoch_losses['reg']/n:.3f}, "
            f"dist={epoch_losses['dist']/n:.3f}) | "
            f"{current_metric_str} | "
            f"{'★ BEST' if is_best else ''} | "
            f"{lr_info} | "
            f"Time: {train_time:.1f}s"
        )

    final_val_str = f"{best_auc:.4f}" if best_metric_type == "auc" else f"{best_metric:.4f}"
    logger.info("=" * 60)
    logger.info("  训练完成!")
    logger.info(
        f"  最优模型: {best_path} | "
        f"best_{best_metric_type}={final_val_str}"
    )
    logger.info("=" * 60)


if __name__ == "__main__":
    main()