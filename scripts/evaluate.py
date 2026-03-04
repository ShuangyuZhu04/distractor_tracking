"""
evaluate.py — 评估入口脚本
==============================
在指定数据集上运行跟踪算法并计算评估指标。

用法:
    python scripts/evaluate.py --config configs/default.yaml --dataset got10k_val --checkpoint checkpoints/best.pth
"""

import argparse
import logging
import os
import sys
import torch
import yaml

# 将项目根目录加入 Python 路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.tracker import Tracker
from data.datasets import build_dataset
from utils.logger import setup_logger
from utils.metrics import Evaluator
from utils.visualizer import TrackingVisualizer

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="目标跟踪算法评估")
    parser.add_argument("--config", type=str, default="configs/default.yaml",
                        help="配置文件路径")
    parser.add_argument("--dataset", type=str, default="otb100",
                        help="数据集名称")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="模型权重文件路径")
    parser.add_argument("--visualize", action="store_true",
                        help="是否生成可视化结果")
    parser.add_argument("--save_dir", type=str, default="results/",
                        help="结果保存目录")
    return parser.parse_args()


def main():
    args = parse_args()

    # ---- 加载配置 ----
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # ---- 初始化日志 ----
    setup_logger(cfg)
    logger.info(f"评估配置: dataset={args.dataset}, config={args.config}")

    # ---- 构建数据集 ----
    dataset = build_dataset(cfg, args.dataset)
    logger.info(f"数据集 [{args.dataset}] 已加载，共 {len(dataset)} 个序列")

    # ---- 构建跟踪器 ----
    tracker = Tracker(cfg)

    # 加载训练好的模型权重
    if args.checkpoint:
        if not os.path.exists(args.checkpoint):
            raise FileNotFoundError(f"找不到权重文件: {args.checkpoint}")

        checkpoint = torch.load(args.checkpoint, map_location=tracker.device)

        # 根据 train.py 的实际保存格式，准确提取 model_state
        if "model_state" in checkpoint:
            state_dict = checkpoint["model_state"]
        elif "model" in checkpoint:
            state_dict = checkpoint["model"]
        else:
            state_dict = checkpoint

        tracker.model.load_state_dict(state_dict)
        tracker.model.eval()   # 加载权重后再次确认 eval 模式
        logger.info(f"✅ 成功加载模型权重: {args.checkpoint}")
    else:
        logger.warning("⚠️ 未提供 checkpoint！将使用随机权重进行评估（AUC 必定极低）。")

    # ---- 构建评估器 & 可视化器 ----
    evaluator = Evaluator(cfg)
    visualizer = TrackingVisualizer(args.save_dir) if args.visualize else None

    # ---- 逐序列评估 ----
    for seq in dataset:
        logger.info(f">>> 评估序列: {seq.name} ({len(seq)} 帧)")
        evaluator.start_sequence(seq.name)

        # 加载首帧并初始化
        first_frame = dataset.load_frame(seq.frame_paths[0])
        init_bbox = seq.ground_truth[0]  # [x, y, w, h]
        tracker.initialize(first_frame, init_bbox)

        # 逐帧跟踪
        for i in range(1, len(seq)):
            frame = dataset.load_frame(seq.frame_paths[i])
            gt_bbox = seq.ground_truth[i]

            evaluator.start_timer()
            result = tracker.track(frame)
            elapsed = evaluator.stop_timer()

            evaluator.update(result["bbox"], gt_bbox, elapsed)

            # 可视化
            if visualizer is not None:
                vis = visualizer.draw_frame(
                    frame=frame,
                    target_bbox=result["bbox"],
                    score=result["score"],
                    distractors=result.get("distractors"),
                    frame_id=i,
                    gt_bbox=gt_bbox,
                )
                visualizer.save_frame(vis, seq.name, i)

        seq_result = evaluator.end_sequence()

        # 生成可视化视频
        if visualizer is not None:
            visualizer.create_video(seq.name)

        tracker.reset()

    # ---- 生成全局报告 ----
    report = evaluator.compute_report()
    evaluator.save_report(args.save_dir)

    logger.info("评估完成")


if __name__ == "__main__":
    main()