"""
demo.py — 跟踪演示脚本
=========================
对单个视频/摄像头进行跟踪演示，实时显示跟踪结果与干扰物信息。

用法:
    # 视频文件
    python scripts/demo.py --config configs/default.yaml --video data/test.mp4

    # 摄像头
    python scripts/demo.py --config configs/default.yaml --camera 0

    # 数据集序列
    python scripts/demo.py --config configs/default.yaml --dataset otb100 --sequence Basketball
"""

import argparse
import logging
import os
import sys

import cv2
import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.tracker import Tracker
from data.datasets import build_dataset
from utils.logger import setup_logger
from utils.visualizer import TrackingVisualizer

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="跟踪演示")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    # 输入源（三选一）
    parser.add_argument("--video", type=str, default=None, help="视频文件路径")
    parser.add_argument("--camera", type=int, default=None, help="摄像头编号")
    parser.add_argument("--dataset", type=str, default=None, help="数据集名称")
    parser.add_argument("--sequence", type=str, default=None, help="数据集中的序列名")
    # 输出
    parser.add_argument("--save_video", action="store_true", help="保存结果视频")
    return parser.parse_args()


def run_on_video(cfg, video_source, save_video=False):
    """在视频或摄像头上运行跟踪演示。"""
    tracker = Tracker(cfg)
    visualizer = TrackingVisualizer()

    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        logger.error(f"无法打开视频源: {video_source}")
        return

    writer = None
    initialized = False

    logger.info(f"演示开始 | 视频源: {video_source}")
    logger.info("请在首帧中用鼠标框选目标区域，按 Enter 确认")

    frame_id = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if not initialized:
            # 首帧：手动框选目标
            bbox = cv2.selectROI("选择目标", frame, fromCenter=False, showCrosshair=True)
            cv2.destroyWindow("选择目标")

            if bbox[2] == 0 or bbox[3] == 0:
                logger.warning("未选择有效目标区域，退出")
                break

            tracker.initialize(frame, np.array(bbox))
            initialized = True
            logger.info(f"目标已初始化: {bbox}")

            if save_video:
                h, w = frame.shape[:2]
                writer = cv2.VideoWriter(
                    "results/demo_output.mp4",
                    cv2.VideoWriter_fourcc(*"mp4v"), 30, (w, h)
                )
        else:
            result = tracker.track(frame)

            vis = visualizer.draw_frame(
                frame=frame,
                target_bbox=result["bbox"],
                score=result["score"],
                distractors=result.get("distractors"),
                frame_id=frame_id,
            )

            cv2.imshow("Tracking Demo", vis)
            if writer:
                writer.write(vis)

        frame_id += 1
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:  # q 或 ESC 退出
            break

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    logger.info("演示结束")


def run_on_dataset(cfg, dataset_name, seq_name):
    """在数据集序列上运行跟踪演示。"""
    dataset = build_dataset(cfg, dataset_name)
    seq = dataset.get_sequence(seq_name)

    if seq is None:
        logger.error(f"序列 '{seq_name}' 不存在，可用序列: {dataset.get_sequence_names()[:10]}...")
        return

    tracker = Tracker(cfg)
    visualizer = TrackingVisualizer()

    # 初始化
    first_frame = dataset.load_frame(seq.frame_paths[0])
    tracker.initialize(first_frame, seq.ground_truth[0])

    logger.info(f"在序列 [{seq_name}] 上运行演示 ({len(seq)} 帧)")

    for i in range(1, len(seq)):
        frame = dataset.load_frame(seq.frame_paths[i])
        result = tracker.track(frame)

        vis = visualizer.draw_frame(
            frame=frame,
            target_bbox=result["bbox"],
            score=result["score"],
            distractors=result.get("distractors"),
            frame_id=i,
            gt_bbox=seq.ground_truth[i],
        )

        cv2.imshow(f"Tracking: {seq_name}", vis)
        key = cv2.waitKey(30) & 0xFF
        if key == ord("q"):
            break

    cv2.destroyAllWindows()


def main():
    args = parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    setup_logger(cfg)

    if args.video:
        run_on_video(cfg, args.video, args.save_video)
    elif args.camera is not None:
        run_on_video(cfg, args.camera, args.save_video)
    elif args.dataset and args.sequence:
        run_on_dataset(cfg, args.dataset, args.sequence)
    else:
        logger.error("请指定输入源: --video / --camera / (--dataset + --sequence)")


if __name__ == "__main__":
    main()
