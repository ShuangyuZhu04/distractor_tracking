"""
metrics.py — 评估指标计算器 (Evaluator)
==========================================
对应结果分析模块

支持的评估指标:
  - Success Plot (AUC):     成功率曲线及其下方面积（主要指标）
  - Precision Plot:         中心误差精确度曲线
  - Normalized Precision:   归一化精确度
  - FPS:                    跟踪帧率

  本项目基于 SiamRPN++ 的单目标跟踪 (SOT) 框架，
  因此使用 SOT 标准指标 (Success/Precision/AUC)，
  而非多目标跟踪 (MOT) 的 MOTA/MOTP 指标。
"""

import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class Evaluator:
    """
    目标跟踪评估器。

    使用流程:
        >>> evaluator = Evaluator(cfg)
        >>> # 对每个序列
        >>> evaluator.start_sequence("Basketball")
        >>> for frame in frames:
        ...     result = tracker.track(frame)
        ...     evaluator.update(result["bbox"], gt_bbox)
        >>> evaluator.end_sequence()
        >>> # 所有序列完成后
        >>> report = evaluator.compute_report()

    参数:
        cfg (dict): 完整配置字典
    """

    # Success Plot 的 IoU 阈值采样点
    IOU_THRESHOLDS = np.arange(0, 1.05, 0.05)  # [0, 0.05, ..., 1.0]

    # Precision Plot 的中心距离阈值采样点
    DIST_THRESHOLDS = np.arange(0, 51, 1)       # [0, 1, ..., 50] 像素

    def __init__(self, cfg: dict = None):
        self.cfg = cfg or {}
        self._sequences_results: Dict[str, Dict] = {}   # 各序列评估结果
        self._current_seq: Optional[str] = None          # 当前正在评估的序列名
        self._current_preds: List[np.ndarray] = []       # 当前序列的预测框
        self._current_gts: List[np.ndarray] = []         # 当前序列的标注框
        self._current_times: List[float] = []            # 当前序列各帧耗时
        self._timer_start: float = 0.0

        logger.info("Evaluator 初始化完成")

    # ==========================================================
    #  序列级操作
    # ==========================================================

    def start_sequence(self, seq_name: str) -> None:
        """
        开始评估一个新的视频序列。

        参数:
            seq_name: 序列名称
        """
        self._current_seq = seq_name
        self._current_preds = []
        self._current_gts = []
        self._current_times = []
        logger.info(f"开始评估序列: {seq_name}")

    def update(
        self,
        pred_bbox: np.ndarray,
        gt_bbox: np.ndarray,
        elapsed_time: Optional[float] = None,
    ) -> None:
        """
        更新单帧评估数据。

        参数:
            pred_bbox (np.ndarray): 预测框 [x, y, w, h]
            gt_bbox (np.ndarray):   标注框 [x, y, w, h]
            elapsed_time (float):   该帧推理耗时（秒），None 则忽略
        """
        self._current_preds.append(np.asarray(pred_bbox, dtype=np.float64))
        self._current_gts.append(np.asarray(gt_bbox, dtype=np.float64))
        if elapsed_time is not None:
            self._current_times.append(elapsed_time)

    def start_timer(self) -> None:
        """启动帧计时器。"""
        self._timer_start = time.perf_counter()

    def stop_timer(self) -> float:
        """停止帧计时器并返回耗时（秒）。"""
        elapsed = time.perf_counter() - self._timer_start
        self._current_times.append(elapsed)
        return elapsed

    def end_sequence(self) -> Dict[str, Any]:
        """
        结束当前序列的评估，计算该序列的指标。

        返回:
            seq_result (dict): 包含该序列的各项指标
        """
        assert self._current_seq is not None, "请先调用 start_sequence()"

        preds = np.array(self._current_preds)
        gts = np.array(self._current_gts)

        # 计算各项指标
        success_curve = self._compute_success_curve(preds, gts)
        precision_curve = self._compute_precision_curve(preds, gts)
        auc = np.mean(success_curve)
        precision_at_20 = precision_curve[20] if len(precision_curve) > 20 else 0.0

        fps = 0.0
        if self._current_times:
            fps = len(self._current_times) / sum(self._current_times)

        seq_result = {
            "sequence_name": self._current_seq,
            "num_frames": len(preds),
            "auc": float(auc),
            "precision_at_20": float(precision_at_20),
            "success_curve": success_curve.tolist(),
            "precision_curve": precision_curve.tolist(),
            "fps": float(fps),
        }

        self._sequences_results[self._current_seq] = seq_result

        logger.info(
            f"序列 [{self._current_seq}] 评估完成 | "
            f"AUC={auc:.4f}, Prec@20={precision_at_20:.4f}, "
            f"FPS={fps:.1f}, 帧数={len(preds)}"
        )

        self._current_seq = None
        return seq_result

    # ==========================================================
    #  全局报告
    # ==========================================================

    def compute_report(self) -> Dict[str, Any]:
        """
        汇总所有序列的评估结果，生成全局报告。

        返回:
            report (dict):
                "overall":    全局平均指标 (AUC, Precision, FPS)
                "per_sequence": 各序列详细指标
                "num_sequences": 序列总数
        """
        if not self._sequences_results:
            logger.warning("无评估数据，请先完成至少一个序列的评估")
            return {}

        all_aucs = [r["auc"] for r in self._sequences_results.values()]
        all_precs = [r["precision_at_20"] for r in self._sequences_results.values()]
        all_fps = [r["fps"] for r in self._sequences_results.values() if r["fps"] > 0]

        report = {
            "num_sequences": len(self._sequences_results),
            "overall": {
                "mean_auc": float(np.mean(all_aucs)),
                "mean_precision_at_20": float(np.mean(all_precs)),
                "mean_fps": float(np.mean(all_fps)) if all_fps else 0.0,
            },
            "per_sequence": dict(self._sequences_results),
        }

        logger.info(
            f"=== 全局评估报告 === | "
            f"序列数: {report['num_sequences']}, "
            f"平均 AUC: {report['overall']['mean_auc']:.4f}, "
            f"平均 Prec@20: {report['overall']['mean_precision_at_20']:.4f}, "
            f"平均 FPS: {report['overall']['mean_fps']:.1f}"
        )

        return report

    def reset(self) -> None:
        """重置评估器（清空所有已累积的结果）。"""
        self._sequences_results.clear()
        self._current_seq = None
        self._current_preds.clear()
        self._current_gts.clear()
        self._current_times.clear()
        logger.info("Evaluator 已重置")

    # ==========================================================
    #  指标计算（静态方法，可独立使用）
    # ==========================================================

    @classmethod
    def _compute_success_curve(
        cls,
        preds: np.ndarray,
        gts: np.ndarray,
    ) -> np.ndarray:
        """
        计算 Success Plot 曲线。

        Success Plot: 对每个 IoU 阈值 t，统计 IoU ≥ t 的帧的比例。
        AUC (Area Under Curve) 即为 Success 曲线的均值。

        参数:
            preds: 预测框, shape = (N, 4), [x, y, w, h]
            gts:   标注框, shape = (N, 4), [x, y, w, h]

        返回:
            curve: Success 曲线, shape = (len(IOU_THRESHOLDS),)
        """
        ious = cls._batch_iou(preds, gts)
        curve = np.array([
            np.mean(ious >= t) for t in cls.IOU_THRESHOLDS
        ])
        return curve

    @classmethod
    def _compute_precision_curve(
        cls,
        preds: np.ndarray,
        gts: np.ndarray,
    ) -> np.ndarray:
        """
        计算 Precision Plot 曲线。

        Precision Plot: 对每个距离阈值 t，
        统计预测框中心与标注框中心距离 ≤ t 的帧的比例。
        通常取 t=20 像素处的值作为精确度指标 (Precision@20)。

        参数:
            preds: 预测框, shape = (N, 4), [x, y, w, h]
            gts:   标注框, shape = (N, 4), [x, y, w, h]

        返回:
            curve: Precision 曲线, shape = (len(DIST_THRESHOLDS),)
        """
        center_dists = cls._center_distance(preds, gts)
        curve = np.array([
            np.mean(center_dists <= t) for t in cls.DIST_THRESHOLDS
        ])
        return curve

    @staticmethod
    def _batch_iou(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
        """
        批量计算 IoU。

        输入格式: [x, y, w, h]
        返回: shape = (N,), 每帧的 IoU 值
        """
        # 转为 [x1, y1, x2, y2]
        a_x1, a_y1 = boxes_a[:, 0], boxes_a[:, 1]
        a_x2, a_y2 = a_x1 + boxes_a[:, 2], a_y1 + boxes_a[:, 3]

        b_x1, b_y1 = boxes_b[:, 0], boxes_b[:, 1]
        b_x2, b_y2 = b_x1 + boxes_b[:, 2], b_y1 + boxes_b[:, 3]

        # 交集
        inter_x1 = np.maximum(a_x1, b_x1)
        inter_y1 = np.maximum(a_y1, b_y1)
        inter_x2 = np.minimum(a_x2, b_x2)
        inter_y2 = np.minimum(a_y2, b_y2)
        inter_area = np.maximum(0, inter_x2 - inter_x1) * np.maximum(0, inter_y2 - inter_y1)

        # 并集
        area_a = boxes_a[:, 2] * boxes_a[:, 3]
        area_b = boxes_b[:, 2] * boxes_b[:, 3]
        union_area = area_a + area_b - inter_area

        iou = np.where(union_area > 0, inter_area / union_area, 0.0)
        return iou

    @staticmethod
    def _center_distance(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
        """
        批量计算中心点距离。

        输入格式: [x, y, w, h]
        返回: shape = (N,), 每帧的中心距离（像素）
        """
        cx_a = boxes_a[:, 0] + boxes_a[:, 2] / 2
        cy_a = boxes_a[:, 1] + boxes_a[:, 3] / 2
        cx_b = boxes_b[:, 0] + boxes_b[:, 2] / 2
        cy_b = boxes_b[:, 1] + boxes_b[:, 3] / 2

        return np.sqrt((cx_a - cx_b) ** 2 + (cy_a - cy_b) ** 2)

    # ==========================================================
    #  结果保存
    # ==========================================================

    def save_report(self, save_dir: str, filename: str = "evaluation_report.txt") -> str:
        """
        将评估报告保存为文本文件。

        参数:
            save_dir:  保存目录
            filename:  文件名

        返回:
            save_path: 报告文件路径
        """
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, filename)

        report = self.compute_report()
        if not report:
            return ""

        with open(save_path, "w", encoding="utf-8") as f:
            f.write("=" * 60 + "\n")
            f.write("  基于背景干扰物挖掘的目标跟踪算法 — 评估报告\n")
            f.write("=" * 60 + "\n\n")

            overall = report["overall"]
            f.write(f"评估序列总数: {report['num_sequences']}\n")
            f.write(f"平均 AUC (Success):     {overall['mean_auc']:.4f}\n")
            f.write(f"平均 Precision@20:      {overall['mean_precision_at_20']:.4f}\n")
            f.write(f"平均 FPS:               {overall['mean_fps']:.1f}\n\n")

            f.write("-" * 60 + "\n")
            f.write(f"{'序列名':<25} {'AUC':>8} {'Prec@20':>8} {'FPS':>8}\n")
            f.write("-" * 60 + "\n")

            for name, result in report["per_sequence"].items():
                f.write(
                    f"{name:<25} {result['auc']:>8.4f} "
                    f"{result['precision_at_20']:>8.4f} "
                    f"{result['fps']:>8.1f}\n"
                )

        logger.info(f"评估报告已保存: {save_path}")
        return save_path
