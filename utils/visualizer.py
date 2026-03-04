"""
visualizer.py — 跟踪结果可视化工具
=====================================
在视频帧上绘制跟踪结果、干扰物标注、注意力热力图等。

可视化元素:
  - 绿色框:  目标跟踪框
  - 红色框:  高相似干扰物 (HIGH_SIMILAR)
  - 橙色框:  遮挡型干扰物 (OCCLUSION)
  - 黄色框:  动态干扰物 (DYNAMIC)
  - 热力图:  CBAM 空间注意力图叠加
"""

import logging
import os
from typing import List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# 风险等级 → 颜色映射 (BGR 格式)
RISK_COLORS = {
    0: (180, 180, 180),   # LOW:          灰色
    1: (0, 255, 255),     # DYNAMIC:      黄色
    2: (0, 165, 255),     # OCCLUSION:    橙色
    3: (0, 0, 255),       # HIGH_SIMILAR: 红色
}

RISK_LABELS = {
    0: "LOW",
    1: "DYN",
    2: "OCC",
    3: "HIGH",
}


class TrackingVisualizer:
    """
    跟踪结果可视化器。

    参数:
        save_dir (str):       可视化结果保存目录
        show_attention (bool): 是否叠加注意力热力图
        show_distractors (bool): 是否绘制干扰物框
        alpha (float):        注意力热力图叠加透明度
    """

    def __init__(
        self,
        save_dir: str = "results/visualizations/",
        show_attention: bool = True,
        show_distractors: bool = True,
        alpha: float = 0.3,
    ):
        self.save_dir = save_dir
        self.show_attention = show_attention
        self.show_distractors = show_distractors
        self.alpha = alpha

        os.makedirs(save_dir, exist_ok=True)
        logger.info(f"TrackingVisualizer 初始化 | save_dir={save_dir}")

    def draw_frame(
        self,
        frame: np.ndarray,
        target_bbox: np.ndarray,
        score: float,
        distractors: Optional[List] = None,
        attention_map: Optional[np.ndarray] = None,
        frame_id: int = 0,
        gt_bbox: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        在帧图像上绘制全部可视化元素。

        参数:
            frame:          原始帧图像 (H, W, 3), BGR
            target_bbox:    目标框 [x, y, w, h]
            score:          跟踪置信度
            distractors:    干扰物列表 (DistractorInfo)
            attention_map:  CBAM 注意力图 (H, W), 值域 [0,1]
            frame_id:       帧号
            gt_bbox:        标注框 (可选，用于对比)

        返回:
            vis_frame: 绘制结果的帧图像
        """
        vis = frame.copy()

        # 1. 注意力热力图叠加
        if self.show_attention and attention_map is not None:
            vis = self._overlay_attention(vis, attention_map)

        # 2. 标注框（蓝色虚线）
        if gt_bbox is not None:
            self._draw_bbox(vis, gt_bbox, color=(255, 0, 0), thickness=1, dashed=True)

        # 3. 干扰物框
        if self.show_distractors and distractors:
            for d in distractors:
                color = RISK_COLORS.get(int(d.risk_level), (180, 180, 180))
                label = RISK_LABELS.get(int(d.risk_level), "?")
                x1, y1, x2, y2 = d.bbox.astype(int)
                w, h = x2 - x1, y2 - y1
                self._draw_bbox(vis, [x1, y1, w, h], color=color, thickness=2)
                cv2.putText(
                    vis, f"{label} {d.similarity:.2f}",
                    (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, color, 1,
                )

        # 4. 目标跟踪框（绿色粗线）
        self._draw_bbox(vis, target_bbox, color=(0, 255, 0), thickness=3)

        # 5. 信息面板
        self._draw_info_panel(vis, frame_id, score, len(distractors or []))

        return vis

    def save_frame(
        self, vis_frame: np.ndarray, seq_name: str, frame_id: int
    ) -> str:
        """保存可视化帧到文件。"""
        seq_dir = os.path.join(self.save_dir, seq_name)
        os.makedirs(seq_dir, exist_ok=True)
        path = os.path.join(seq_dir, f"{frame_id:06d}.jpg")
        cv2.imwrite(path, vis_frame)
        return path

    def create_video(
        self,
        seq_name: str,
        fps: float = 30.0,
    ) -> str:
        """
        将保存的可视化帧合成为视频文件。

        参数:
            seq_name: 序列名称
            fps:      视频帧率

        返回:
            video_path: 输出视频路径
        """
        seq_dir = os.path.join(self.save_dir, seq_name)
        frames = sorted([
            os.path.join(seq_dir, f) for f in os.listdir(seq_dir)
            if f.endswith(".jpg")
        ])

        if not frames:
            logger.warning(f"序列 {seq_name} 无可视化帧")
            return ""

        # 读取第一帧获取尺寸
        sample = cv2.imread(frames[0])
        H, W = sample.shape[:2]

        video_path = os.path.join(self.save_dir, f"{seq_name}_result.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(video_path, fourcc, fps, (W, H))

        for fp in frames:
            writer.write(cv2.imread(fp))
        writer.release()

        logger.info(f"可视化视频已生成: {video_path}")
        return video_path

    # ==========================================================
    #  内部绘制方法
    # ==========================================================

    @staticmethod
    def _draw_bbox(
        img: np.ndarray,
        bbox: np.ndarray,
        color: Tuple[int, int, int] = (0, 255, 0),
        thickness: int = 2,
        dashed: bool = False,
    ) -> None:
        """绘制边界框 [x, y, w, h]。"""
        x, y, w, h = [int(v) for v in bbox]
        if dashed:
            # 虚线绘制
            for i in range(0, w, 10):
                cv2.line(img, (x + i, y), (x + min(i + 5, w), y), color, thickness)
                cv2.line(img, (x + i, y + h), (x + min(i + 5, w), y + h), color, thickness)
            for i in range(0, h, 10):
                cv2.line(img, (x, y + i), (x, y + min(i + 5, h)), color, thickness)
                cv2.line(img, (x + w, y + i), (x + w, y + min(i + 5, h)), color, thickness)
        else:
            cv2.rectangle(img, (x, y), (x + w, y + h), color, thickness)

    def _overlay_attention(
        self, img: np.ndarray, attention_map: np.ndarray
    ) -> np.ndarray:
        """将 CBAM 注意力图作为热力图叠加到原始帧上。"""
        H, W = img.shape[:2]
        attn = cv2.resize(attention_map, (W, H))
        attn = (attn * 255).astype(np.uint8)
        heatmap = cv2.applyColorMap(attn, cv2.COLORMAP_JET)
        return cv2.addWeighted(img, 1 - self.alpha, heatmap, self.alpha, 0)

    @staticmethod
    def _draw_info_panel(
        img: np.ndarray,
        frame_id: int,
        score: float,
        num_distractors: int,
    ) -> None:
        """在左上角绘制半透明信息面板。"""
        overlay = img.copy()
        cv2.rectangle(overlay, (5, 5), (280, 70), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)

        cv2.putText(
            img, f"Frame: {frame_id}  Score: {score:.3f}",
            (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
        )
        cv2.putText(
            img, f"Distractors: {num_distractors}",
            (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
        )
