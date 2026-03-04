"""
check_train_data.py — 训练数据健全性检查 (Sanity Check)
=========================================================
在投入长时间训练之前，可视化验证以下关键环节的正确性：

  1. 模板/搜索图像裁剪与归一化是否正常
  2. GT bbox 经坐标变换后在搜索图中的位置是否准确
  3. 正样本 anchor 的回归目标解码后是否与 GT 重合
  4. 干扰物掩码标记的 anchor 是否合理（位于目标周围的模糊区域）

用法:
    python scripts/check_train_data.py --config configs/default.yaml
    python scripts/check_train_data.py --config configs/default.yaml --num_samples 8

输出:
    results/sanity_check/
    ├── sample_0.png      # 模板 + 搜索（标注框）对比图
    ├── sample_1.png
    ├── ...
    └── anchor_stats.txt  # anchor 标签分布统计
"""

import argparse
import logging
import os
import sys

import cv2
import matplotlib
matplotlib.use("Agg")  
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.anchor import AnchorTargetGenerator
from data.datasets import build_dataset
from data.train_dataset import TrackingTrainDataset, IMAGENET_MEAN, IMAGENET_STD

logger = logging.getLogger(__name__)


# ============================================================
#  工具函数
# ============================================================

def denormalize_tensor(tensor_chw: np.ndarray) -> np.ndarray:
    """
    将 ImageNet 归一化的 (3, H, W) float32 张量还原为 (H, W, 3) uint8 RGB 图像。

    逆操作: img = tensor * std + mean, 然后 clip 到 [0, 255]
    """
    img = tensor_chw.transpose(1, 2, 0)  # CHW → HWC
    img = img * IMAGENET_STD + IMAGENET_MEAN
    img = np.clip(img * 255, 0, 255).astype(np.uint8)
    return img


def decode_regression(
    anchor: np.ndarray, reg: np.ndarray
) -> np.ndarray:
    """
    从 anchor 和回归偏移量解码出预测框。

    逆公式:
      pred_cx = dx * anchor_w + anchor_cx
      pred_cy = dy * anchor_h + anchor_cy
      pred_w  = exp(dw) * anchor_w
      pred_h  = exp(dh) * anchor_h

    参数:
        anchor: (4,) 或 (N, 4) [cx, cy, w, h]
        reg:    (4,) 或 (N, 4) [dx, dy, dw, dh]

    返回:
        decoded: 同 shape, [cx, cy, w, h]
    """
    if anchor.ndim == 1:
        anchor = anchor[np.newaxis, :]
        reg = reg[np.newaxis, :]
        squeeze = True
    else:
        squeeze = False

    pred_cx = reg[:, 0] * anchor[:, 2] + anchor[:, 0]
    pred_cy = reg[:, 1] * anchor[:, 3] + anchor[:, 1]
    pred_w = np.exp(reg[:, 2]) * anchor[:, 2]
    pred_h = np.exp(reg[:, 3]) * anchor[:, 3]

    decoded = np.stack([pred_cx, pred_cy, pred_w, pred_h], axis=1)
    return decoded[0] if squeeze else decoded


def cxcywh_to_xyxy(bbox: np.ndarray) -> tuple:
    """[cx, cy, w, h] → (x1, y1, x2, y2)"""
    cx, cy, w, h = bbox[:4]
    return cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2


# ============================================================
#  主可视化函数
# ============================================================

def visualize_sample(
    template: np.ndarray,
    search: np.ndarray,
    cls_label: np.ndarray,
    reg_label: np.ndarray,
    distractor_mask: np.ndarray,
    anchors: np.ndarray,
    save_path: str,
    sample_idx: int,
):
    """
    绘制一个训练样本的完整可视化诊断图。

    布局 (1 行 3 列):
      [模板图]  [搜索图 + GT 框 + 正样本 anchor]  [搜索图 + 干扰物 anchor]

    颜色编码:
      绿色实线   — 从正样本 anchor 回归解码出的 GT 框
      蓝色虚线   — 正样本 anchor 原始框（解码前）
      红色虚线   — 干扰物掩码标记的 anchor 框
      黄色点     — 所有 anchor 中心点（淡色背景）
    """
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    fig.suptitle(
        f"Sample #{sample_idx}  |  "
        f"正样本={int((cls_label == 1).sum())}  "
        f"负样本={int((cls_label == 0).sum())}  "
        f"忽略={int((cls_label == -1).sum())}  "
        f"干扰物anchor={int((distractor_mask > 0).sum())}",
        fontsize=13, fontweight="bold",
    )

    # ---- 列 1: 模板图 ----
    ax = axes[0]
    ax.imshow(template)
    ax.set_title("Template (127×127)", fontsize=11)
    ax.axis("off")

    # ---- 列 2: 搜索图 + GT + 正样本 anchor ----
    ax = axes[1]
    ax.imshow(search)
    ax.set_title("Search + GT (绿) + 正样本anchor (蓝)", fontsize=11)

    # 正样本 anchor 索引
    pos_indices = np.where(cls_label == 1)[0]

    # 画正样本 anchor 原始框（蓝色虚线）
    for idx in pos_indices:
        x1, y1, x2, y2 = cxcywh_to_xyxy(anchors[idx])
        rect = patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=1.0, edgecolor="dodgerblue", facecolor="none",
            linestyle="--", alpha=0.7,
        )
        ax.add_patch(rect)

    # 从正样本回归目标解码出 GT 框（所有正样本应解码到同一位置）
    if len(pos_indices) > 0:
        # reg_label: (4, N), anchors: (N, 4)
        pos_anchor = anchors[pos_indices[0]]
        pos_reg = reg_label[:, pos_indices[0]]  # (4,)
        decoded_gt = decode_regression(pos_anchor, pos_reg)
        x1, y1, x2, y2 = cxcywh_to_xyxy(decoded_gt)
        rect = patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=2.5, edgecolor="lime", facecolor="none",
            linestyle="-", label="Decoded GT",
        )
        ax.add_patch(rect)

        # 验证: 用多个正样本解码，检查一致性
        if len(pos_indices) >= 2:
            decoded_all = decode_regression(
                anchors[pos_indices],
                reg_label[:, pos_indices].T,
            )
            spread = decoded_all.std(axis=0)
            consistency_ok = spread.max() < 2.0  # 允许 2 像素内浮动
            color = "lime" if consistency_ok else "red"
            ax.text(
                5, search.shape[0] - 10,
                f"GT一致性: {'✓' if consistency_ok else '✗'} "
                f"(std={spread.max():.2f}px)",
                fontsize=9, color=color,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.6),
            )

    ax.legend(fontsize=9, loc="upper right")
    ax.axis("off")

    # ---- 列 3: 搜索图 + 干扰物 anchor ----
    ax = axes[2]
    ax.imshow(search)
    ax.set_title("Search + 干扰物anchor (红)", fontsize=11)

    # 先淡淡画出所有 anchor 中心（帮助理解网格覆盖）
    ax.scatter(
        anchors[:, 0], anchors[:, 1],
        s=0.3, c="yellow", alpha=0.15, zorder=1,
    )

    # 干扰物 anchor（红色虚线）
    dist_indices = np.where(distractor_mask > 0)[0]
    for idx in dist_indices:
        x1, y1, x2, y2 = cxcywh_to_xyxy(anchors[idx])
        rect = patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=1.2, edgecolor="red", facecolor="red",
            linestyle="--", alpha=0.25, zorder=2,
        )
        ax.add_patch(rect)

    # 重新画 GT 框做对比（绿色）
    if len(pos_indices) > 0:
        x1, y1, x2, y2 = cxcywh_to_xyxy(decoded_gt)
        rect = patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=2.5, edgecolor="lime", facecolor="none", zorder=3,
        )
        ax.add_patch(rect)

    # 标注干扰物 anchor 的 IoU 范围
    if len(dist_indices) > 0 and len(pos_indices) > 0:
        # 计算干扰物 anchor 与 decoded GT 的 IoU
        dist_ious = AnchorTargetGenerator._compute_iou(
            anchors[dist_indices], decoded_gt
        )
        ax.text(
            5, 15,
            f"干扰物anchor IoU: [{dist_ious.min():.2f}, {dist_ious.max():.2f}]",
            fontsize=9, color="red",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.6),
        )

    ax.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ============================================================
#  主函数
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="训练数据健全性检查")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--num_samples", type=int, default=4,
                        help="可视化样本数量")
    args = parser.parse_args()

    import yaml
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # ---- 输出目录 ----
    out_dir = "results/sanity_check"
    os.makedirs(out_dir, exist_ok=True)

    # ---- 加载数据集 ----
    logger.info("加载数据集...")
    train_datasets = []
    for name in cfg["train"].get("datasets", ["otb100"]):
        try:
            ds = build_dataset(cfg, name)
            if len(ds) > 0:
                train_datasets.append(ds)
                logger.info(f"  [{name}] {len(ds)} 个序列")
        except Exception as e:
            logger.warning(f"  [{name}] 加载失败: {e}")

    if not train_datasets:
        logger.error(
            "没有可用数据集! 请确保 data/OTB100/ 目录存在且包含序列。\n"
            "  提示: 至少准备一个序列用于 sanity check, 例如:\n"
            "    data/OTB100/Basketball/img/0001.jpg ... \n"
            "    data/OTB100/Basketball/groundtruth_rect.txt"
        )
        sys.exit(1)

    # ---- 构建训练数据集 ----
    train_dataset = TrackingTrainDataset.from_config(cfg, train_datasets)

    logger.info(f"训练数据集: {len(train_dataset)} 个帧对")
    logger.info(f"Anchor 网格: {train_dataset.anchor_gen.total_anchors} 个 anchor")

    # 获取预生成的 anchor 坐标（用于可视化）
    anchors = train_dataset.anchor_gen.anchors  # (1445, 4)

    # ---- 逐样本检查 ----
    num_check = min(args.num_samples, len(train_dataset))
    stats_lines = []

    stats_lines.append("=" * 70)
    stats_lines.append("  训练数据健全性检查报告")
    stats_lines.append("=" * 70)
    stats_lines.append(f"数据集序列数: {sum(len(ds) for ds in train_datasets)}")
    stats_lines.append(f"帧对总数:     {len(train_dataset)}")
    stats_lines.append(f"Anchor 总数:  {train_dataset.anchor_gen.total_anchors}")
    stats_lines.append(f"检查样本数:   {num_check}")
    stats_lines.append("-" * 70)

    for i in range(num_check):
        logger.info(f"处理样本 {i}/{num_check}...")

        sample = train_dataset[i]

        template_np = denormalize_tensor(sample["template"].numpy())
        search_np = denormalize_tensor(sample["search"].numpy())
        cls_label = sample["cls_label"].numpy()       # (N,)
        reg_label = sample["reg_label"].numpy()       # (4, N)
        dist_mask = sample["distractor_mask"].numpy()  # (N,)

        # 统计
        n_pos = int((cls_label == 1).sum())
        n_neg = int((cls_label == 0).sum())
        n_ign = int((cls_label == -1).sum())
        n_dist = int((dist_mask > 0).sum())

        stats_lines.append(
            f"Sample {i}: 正={n_pos:3d}  负={n_neg:3d}  "
            f"忽略={n_ign:4d}  干扰物={n_dist:3d}  "
            f"采样总数={n_pos + n_neg}"
        )

        # 回归目标统计（仅正样本）
        if n_pos > 0:
            pos_idx = np.where(cls_label == 1)[0]
            pos_regs = reg_label[:, pos_idx]  # (4, n_pos)
            stats_lines.append(
                f"         回归目标均值: dx={pos_regs[0].mean():.3f}, "
                f"dy={pos_regs[1].mean():.3f}, "
                f"dw={pos_regs[2].mean():.3f}, "
                f"dh={pos_regs[3].mean():.3f}"
            )

            # 解码一致性检查
            decoded = decode_regression(
                anchors[pos_idx], reg_label[:, pos_idx].T
            )
            spread = decoded.std(axis=0)
            stats_lines.append(
                f"         GT解码一致性: "
                f"cx_std={spread[0]:.2f}, cy_std={spread[1]:.2f}, "
                f"w_std={spread[2]:.2f}, h_std={spread[3]:.2f}"
            )

        # 绘图
        save_path = os.path.join(out_dir, f"sample_{i}.png")
        visualize_sample(
            template_np, search_np,
            cls_label, reg_label, dist_mask,
            anchors, save_path, i,
        )
        logger.info(f"  已保存: {save_path}")

    # ---- 全局 anchor 网格统计 ----
    stats_lines.append("-" * 70)
    stats_lines.append("Anchor 网格统计:")
    stats_lines.append(
        f"  中心 X 范围: [{anchors[:, 0].min():.1f}, {anchors[:, 0].max():.1f}]"
    )
    stats_lines.append(
        f"  中心 Y 范围: [{anchors[:, 1].min():.1f}, {anchors[:, 1].max():.1f}]"
    )
    unique_w = np.unique(anchors[:, 2])
    unique_h = np.unique(anchors[:, 3])
    for w, h in zip(unique_w, unique_h):
        stats_lines.append(f"  Anchor 尺寸: {w:.1f} × {h:.1f} (ratio={w/h:.2f})")

    stats_lines.append("=" * 70)

    # 写入统计文件
    stats_path = os.path.join(out_dir, "anchor_stats.txt")
    with open(stats_path, "w", encoding="utf-8") as f:
        f.write("\n".join(stats_lines))

    # 同时打印到控制台
    print("\n".join(stats_lines))

    logger.info(f"\n全部检查完成! 结果保存在: {out_dir}/")
    logger.info("请检查:")
    logger.info("  1. 模板图像是否清晰、目标居中")
    logger.info("  2. 绿色 GT 框是否准确框住搜索图中的目标")
    logger.info("  3. 蓝色正样本 anchor 是否分布在目标附近")
    logger.info("  4. 红色干扰物 anchor 是否在目标周围（非重叠、非远离）")
    logger.info("  5. anchor_stats.txt 中 GT 解码一致性 std 是否接近 0")


if __name__ == "__main__":
    main()