"""
datasets.py — 数据集加载器 (DataLoader)
==========================================
支持主流目标跟踪数据集的统一加载

设计原则:
  所有数据集统一为相同的输出接口:
    - 视频序列列表
    - 每帧图像路径
    - 目标标注框 (ground truth bbox)
  上层调用方无需关心数据集内部差异。
"""

import logging
import os
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# ============================================================
#  序列信息数据类
# ============================================================

class SequenceInfo:
    """
    单个视频序列的完整信息。

    属性:
        name (str):             序列名称（如 "Basketball", "Car1"）
        frame_paths (list):     各帧图像文件路径列表
        ground_truth (ndarray): 目标标注框, shape = (N_frames, 4)
                                格式: [x, y, w, h]（左上角 + 宽高）
        attrs (list):           序列属性标签（如 ["IV", "OCC", "DEF"]）
        img_size (tuple):       图像尺寸 (H, W)
    """

    def __init__(
        self,
        name: str,
        frame_paths: List[str],
        ground_truth: np.ndarray,
        attrs: Optional[List[str]] = None,
        img_size: Optional[Tuple[int, int]] = None,
    ):
        self.name = name
        self.frame_paths = frame_paths
        self.ground_truth = ground_truth
        self.attrs = attrs or []
        self.img_size = img_size

    def __len__(self) -> int:
        return len(self.frame_paths)

    def __repr__(self) -> str:
        return (
            f"SequenceInfo(name='{self.name}', "
            f"frames={len(self)}, attrs={self.attrs})"
        )


# ============================================================
#  数据集基类
# ============================================================

class BaseTrackingDataset(ABC):
    """
    跟踪数据集基类 — 定义统一加载接口。

    子类需实现:
        _load_sequences(): 解析数据集目录结构，返回 SequenceInfo 列表
    """

    def __init__(self, root: str, name: str = "unknown"):
        self.root = Path(root)
        self.name = name
        self.sequences: List[SequenceInfo] = []

        if not self.root.exists():
            logger.warning(f"数据集根目录不存在: {self.root}")
        else:
            self.sequences = self._load_sequences()
            logger.info(
                f"数据集 [{self.name}] 加载完成 | "
                f"序列数: {len(self.sequences)}, "
                f"路径: {self.root}"
            )

    @abstractmethod
    def _load_sequences(self) -> List[SequenceInfo]:
        """解析数据集目录，返回序列列表。子类必须实现。"""
        ...

    def get_sequence(self, name: str) -> Optional[SequenceInfo]:
        """按名称查找序列。"""
        for seq in self.sequences:
            if seq.name == name:
                return seq
        logger.warning(f"序列 '{name}' 未找到")
        return None

    def get_sequence_names(self) -> List[str]:
        """获取所有序列名称。"""
        return [seq.name for seq in self.sequences]

    def load_frame(self, frame_path: str) -> np.ndarray:
        """
        加载单帧图像。

        参数:
            frame_path: 帧图像文件路径

        返回:
            image: BGR 格式图像, shape = (H, W, 3)
        """
        img = cv2.imread(frame_path)
        if img is None:
            raise FileNotFoundError(f"无法加载图像: {frame_path}")
        return img

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> SequenceInfo:
        return self.sequences[idx]

    def __iter__(self):
        return iter(self.sequences)


# ============================================================
#  OTB100 数据集
# ============================================================

class OTB100Dataset(BaseTrackingDataset):
    """
    OTB100 数据集加载器。

    目录结构:
        OTB100/
        ├── Basketball/
        │   ├── img/
        │   │   ├── 0001.jpg
        │   │   ├── 0002.jpg
        │   │   └── ...
        │   └── groundtruth_rect.txt
        ├── Biker/
        │   ├── img/
        │   │   └── ...
        │   └── groundtruth_rect.txt
        └── ...

    标注格式: 每行一帧, x,y,w,h (逗号分隔)

    属性标签:
        IV  - 光照变化       OCC - 遮挡
        DEF - 形变           MB  - 运动模糊
        FM  - 快速运动       IPR - 面内旋转
        OPR - 面外旋转       OV  - 出视野
        BC  - 背景杂波       LR  - 低分辨率
        SV  - 尺度变化
    """
    # OTB100 序列属性标签映射（部分示例）
    ATTRS_MAP = {
        "Basketball": ["IV", "OCC", "DEF", "OPR", "BC"],
        "Biker": ["SV", "OCC", "MB", "FM", "OPR", "OV", "LR"],
        "Bird1": ["DEF", "FM", "OV"],
        "Bird2": ["OCC", "DEF", "FM", "IPR", "OPR"],
        "BlurBody": ["SV", "DEF", "MB", "FM", "IPR"],
        "BlurCar1": ["MB", "FM"],
        "BlurCar2": ["SV", "MB", "FM"],
        "BlurCar3": ["MB", "FM"],
        "BlurCar4": ["MB", "FM"],
        "BlurFace": ["MB", "FM", "IPR"],
        "BlurOwl": ["SV", "MB", "FM", "IPR"],
        "Board": ["SV", "MB", "FM", "OPR", "OV", "BC"],
        "Bolt": ["OCC", "DEF", "IPR", "OPR"],
        "Bolt2": ["DEF", "BC"],
        "Box": ["IV", "SV", "OCC", "MB", "IPR", "OPR", "OV", "BC", "LR"],
        "Boy": ["SV", "MB", "FM", "IPR", "OPR"],
        "Car1": ["IV", "SV", "MB", "FM", "BC", "LR"],
        "Car2": ["IV", "SV", "MB", "FM", "BC"],
        "Car24": ["IV", "SV", "BC"],
        "Car4": ["IV", "SV"],
        "CarDark": ["IV", "BC"],
        "CarScale": ["SV", "OCC", "FM", "IPR", "OPR"],
        "ClifBar": ["SV", "OCC", "MB", "FM", "IPR", "OV", "BC"],
        "Coke": ["IV", "OCC", "FM", "IPR", "OPR", "BC"],
        "Couple": ["SV", "DEF", "FM", "OPR", "BC"],
        "Coupon": ["OCC", "BC"],
        "Crossing": ["SV", "DEF", "FM", "OPR", "BC"],
        "Crowds": ["IV", "DEF", "BC"],
        "Dancer": ["SV", "DEF", "IPR", "OPR"],
        "Dancer2": ["DEF"],
        "David": ["IV", "SV", "OCC", "DEF", "MB", "IPR", "OPR"],
        "David2": ["IPR", "OPR"],
        "David3": ["OCC", "DEF", "OPR", "BC"],
        "Deer": ["MB", "FM", "IPR", "BC", "LR"],
        "Diving": ["SV", "DEF", "IPR"],
        "Dog": ["SV", "DEF", "OPR"],
        "Dog1": ["SV", "IPR", "OPR"],
        "Doll": ["IV", "SV", "OCC", "IPR", "OPR"],
        "DragonBaby": ["SV", "OCC", "MB", "FM", "IPR", "OPR", "OV"],
        "Dudek": ["SV", "OCC", "DEF", "FM", "IPR", "OPR", "OV", "BC"],
        "FaceOcc1": ["OCC"],
        "FaceOcc2": ["IV", "OCC", "IPR", "OPR"],
        "Fish": ["IV"],
        "FleetFace": ["SV", "DEF", "MB", "FM", "IPR", "OPR"],
        "Football": ["OCC", "IPR", "OPR", "BC"],
        "Football1": ["IPR", "OPR", "BC"],
        "Freeman1": ["SV", "IPR", "OPR"],
        "Freeman3": ["SV", "IPR", "OPR"],
        "Freeman4": ["SV", "OCC", "IPR", "OPR"],
        "Girl": ["SV", "OCC", "IPR", "OPR"],
        "Girl2": ["SV", "OCC", "DEF", "MB", "OPR"],
        "Gym": ["SV", "DEF", "IPR", "OPR"],
        "Human2": ["IV", "SV", "MB", "OPR"],
        "Human3": ["SV", "OCC", "DEF", "OPR", "BC"],
        "Human4-2": ["IV", "SV", "OCC", "DEF"],
        "Human5": ["SV", "OCC", "DEF"],
        "Human6": ["SV", "OCC", "DEF", "FM", "OPR", "OV"],
        "Human7": ["IV", "SV", "OCC", "DEF", "MB", "FM"],
        "Human8": ["IV", "SV", "DEF"],
        "Human9": ["IV", "SV", "DEF", "MB", "FM"],
        "Ironman": ["IV", "SV", "OCC", "MB", "FM", "IPR", "OPR", "OV", "BC", "LR"],
        "Jogging-1": ["OCC", "DEF", "OPR"],
        "Jogging-2": ["OCC", "DEF", "OPR"],
        "Jump": ["SV", "OCC", "DEF", "MB", "FM", "IPR", "OPR"],
        "Jumping": ["MB", "FM"],
        "KiteSurf": ["IV", "OCC", "IPR", "OPR"],
        "Lemming": ["IV", "SV", "OCC", "FM", "OPR", "OV"],
        "Liquor": ["IV", "SV", "OCC", "MB", "FM", "OPR", "OV", "BC"],
        "Man": ["IV"],
        "Matrix": ["IV", "SV", "OCC", "FM", "IPR", "OPR", "BC"],
        "Mhyang": ["IV", "DEF", "OPR", "BC"],
        "MotorRolling": ["IV", "SV", "MB", "FM", "IPR", "BC", "LR"],
        "MountainBike": ["IPR", "OPR", "BC"],
        "Panda": ["SV", "OCC", "DEF", "IPR", "OPR", "OV", "LR"],
        "RedTeam": ["SV", "OCC", "IPR", "OPR", "LR"],
        "Rubik": ["SV", "OCC", "IPR", "OPR"],
        "Shaking": ["IV", "SV", "IPR", "OPR", "BC"],
        "Singer1": ["IV", "SV", "OCC", "OPR"],
        "Singer2": ["IV", "DEF", "IPR", "OPR", "BC"],
        "Skater": ["SV", "DEF", "IPR", "OPR"],
        "Skater2": ["SV", "DEF", "FM", "IPR", "OPR"],
        "Skating1": ["IV", "SV", "OCC", "DEF", "OPR", "BC"],
        "Skating2-1": ["SV", "OCC", "DEF", "FM", "OPR"],
        "Skating2-2": ["SV", "OCC", "DEF", "FM", "OPR"],
        "Skiing": ["IV", "SV", "DEF", "IPR", "OPR"],
        "Soccer": ["IV", "SV", "OCC", "MB", "FM", "IPR", "OPR", "BC"],
        "Subway": ["OCC", "DEF", "BC"],
        "Surfer": ["SV", "FM", "IPR", "OPR", "LR"],
        "Suv": ["OCC", "IPR", "OV"],
        "Sylvester": ["IV", "IPR", "OPR"],
        "Tiger1": ["IV", "OCC", "DEF", "MB", "FM", "IPR", "OPR"],
        "Tiger2": ["IV", "OCC", "DEF", "MB", "FM", "IPR", "OPR", "OV"],
        "Toy": ["SV", "FM", "IPR", "OPR"],
        "Trans": ["IV", "SV", "OCC", "DEF"],
        "Trellis": ["IV", "SV", "IPR", "OPR", "BC"],
        "Twinnings": ["SV", "OPR"],
        "Vase": ["SV", "FM", "IPR"],
        "Walking": ["SV", "OCC", "DEF"],
        "Walking2": ["SV", "OCC", "LR"],
        "Woman": ["IV", "SV", "OCC", "DEF", "MB", "FM", "OPR"],
    }

    def __init__(self, root: str):
        super().__init__(root, name="OTB100")

    def _load_sequences(self) -> List[SequenceInfo]:
        sequences = []

        for seq_dir in sorted(self.root.iterdir()):
            if not seq_dir.is_dir():
                continue

            seq_name = seq_dir.name
            img_dir = seq_dir / "img"

            if not img_dir.exists():
                logger.debug(f"跳过不完整序列: {seq_name}")
                continue

            # 加载帧路径（按文件名中的数字排序，非常关键）
            frame_paths = sorted(
                [str(p) for p in img_dir.glob("*.jpg")],
                key=lambda x: int(Path(x).stem)
            )

            if not frame_paths:
                continue

            # 处理 Jogging, Skating2 等多目标序列
            gt_files = list(seq_dir.glob("groundtruth_rect*.txt"))
            if not gt_files:
                continue

            for gt_file in gt_files:
                # 区分多目标的子序列名称
                sub_name = seq_name
                if gt_file.name == "groundtruth_rect.1.txt":
                    sub_name = f"{seq_name}-1"
                elif gt_file.name == "groundtruth_rect.2.txt":
                    sub_name = f"{seq_name}-2"

                # 加载标注文件
                ground_truth = self._parse_gt_file(gt_file)

                # 确保帧数与标注数匹配
                min_len = min(len(frame_paths), len(ground_truth))
                seq_frames = frame_paths[:min_len]
                seq_gt = ground_truth[:min_len]

                attrs = self.ATTRS_MAP.get(seq_name, [])

                sequences.append(SequenceInfo(
                    name=sub_name,
                    frame_paths=seq_frames,
                    ground_truth=seq_gt,
                    attrs=attrs,
                ))

        return sequences

    @staticmethod
    def _parse_gt_file(gt_path: Path) -> np.ndarray:
        """
        用正则表达式处理 OTB 的分隔符。
        """
        gt_list = []
        with open(gt_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                # 正则分割：匹配任何连续的逗号、制表符或空格
                values = re.split(r'[,\t\s]+', line)
                try:
                    gt_list.append([float(v) for v in values[:4]])
                except ValueError:
                    continue

        return np.array(gt_list, dtype=np.float64)


# ============================================================
#  UAV123 数据集
# ============================================================

class UAV123Dataset(BaseTrackingDataset):
    """
    UAV123 数据集加载器。

    目录结构:
        UAV123/
        ├── data_seq/
        │   └── UAV123/
        │       ├── bike1/
        │       │   ├── 0000001.jpg
        │       │   └── ...
        │       └── ...
        └── anno/
            └── UAV123/
                ├── bike1.txt
                └── ...

    标注格式: 每行一帧, x,y,w,h (逗号分隔)
    """

    def __init__(self, root: str):
        super().__init__(root, name="UAV123")

    def _load_sequences(self) -> List[SequenceInfo]:
        """解析 UAV123 目录结构。"""
        sequences = []
        seq_dir = self.root / "data_seq" / "UAV123"
        anno_dir = self.root / "anno" / "UAV123"

        if not seq_dir.exists():
            logger.warning(f"UAV123 序列目录不存在: {seq_dir}")
            return sequences

        for seq_folder in sorted(seq_dir.iterdir()):
            if not seq_folder.is_dir():
                continue

            seq_name = seq_folder.name
            gt_file = anno_dir / f"{seq_name}.txt"

            if not gt_file.exists():
                logger.debug(f"跳过无标注序列: {seq_name}")
                continue

            # 加载帧路径
            frame_paths = sorted(
                [str(p) for p in seq_folder.glob("*.jpg")],
                key=lambda x: int(Path(x).stem)
            )

            if not frame_paths:
                continue

            # 加载标注
            ground_truth = self._parse_gt_file(gt_file)

            min_len = min(len(frame_paths), len(ground_truth))
            frame_paths = frame_paths[:min_len]
            ground_truth = ground_truth[:min_len]

            sequences.append(SequenceInfo(
                name=seq_name,
                frame_paths=frame_paths,
                ground_truth=ground_truth,
            ))

        return sequences

    @staticmethod
    def _parse_gt_file(gt_path: Path) -> np.ndarray:
        """解析 UAV123 标注文件（逗号分隔，NaN 表示缺失帧）。"""
        gt_list = []
        with open(gt_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line or "NaN" in line:
                    gt_list.append([0, 0, 0, 0])  # 缺失帧标记
                    continue
                values = line.split(",")
                try:
                    gt_list.append([float(v) for v in values[:4]])
                except ValueError:
                    gt_list.append([0, 0, 0, 0])

        return np.array(gt_list, dtype=np.float64)


# ============================================================
#  VOT2022 数据集
# ============================================================

class VOT2022Dataset(BaseTrackingDataset):
    """
    VOT2022 数据集加载器。

    目录结构:
        VOT2022/
        ├── sequences/
        │   ├── agility/
        │   │   ├── color/
        │   │   │   ├── 00000001.jpg
        │   │   │   └── ...
        │   │   └── groundtruth.txt
        │   └── ...
        └── list.txt  (可选: 序列名列表)

    标注格式: 多边形标注, 每行 8 个值 x1,y1,x2,y2,x3,y3,x4,y4
             本加载器将多边形转为外接矩形 [x, y, w, h]
    """

    def __init__(self, root: str):
        super().__init__(root, name="VOT2022")

    def _load_sequences(self) -> List[SequenceInfo]:
        """解析 VOT2022 目录结构。"""
        sequences = []
        seq_base = self.root / "sequences"

        if not seq_base.exists():
            logger.warning(f"VOT2022 序列目录不存在: {seq_base}")
            return sequences

        for seq_folder in sorted(seq_base.iterdir()):
            if not seq_folder.is_dir():
                continue

            seq_name = seq_folder.name
            color_dir = seq_folder / "color"
            gt_file = seq_folder / "groundtruth.txt"

            if not color_dir.exists() or not gt_file.exists():
                continue

            # 加载帧路径
            frame_paths = sorted(
                [str(p) for p in color_dir.glob("*.jpg")],
                key=lambda x: int(Path(x).stem)
            )

            if not frame_paths:
                continue

            # 加载并转换标注（多边形 → 外接矩形）
            ground_truth = self._parse_vot_gt(gt_file)

            min_len = min(len(frame_paths), len(ground_truth))
            frame_paths = frame_paths[:min_len]
            ground_truth = ground_truth[:min_len]

            sequences.append(SequenceInfo(
                name=seq_name,
                frame_paths=frame_paths,
                ground_truth=ground_truth,
            ))

        return sequences

    @staticmethod
    def _parse_vot_gt(gt_path: Path) -> np.ndarray:
        """
        解析 VOT 多边形标注并转为外接矩形。

        VOT 格式: x1,y1,x2,y2,x3,y3,x4,y4 (四个顶点坐标)
        输出格式: x,y,w,h (最小外接矩形)
        """
        gt_list = []
        with open(gt_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                values = line.split(",")
                try:
                    coords = [float(v) for v in values[:8]]
                    # 提取 x 和 y 坐标
                    xs = coords[0::2]  # x1, x2, x3, x4
                    ys = coords[1::2]  # y1, y2, y3, y4
                    x_min, x_max = min(xs), max(xs)
                    y_min, y_max = min(ys), max(ys)
                    gt_list.append([x_min, y_min, x_max - x_min, y_max - y_min])
                except (ValueError, IndexError):
                    gt_list.append([0, 0, 0, 0])

        return np.array(gt_list, dtype=np.float64)


# ============================================================
#  GOT-10k 数据集 (SiamRPN++ 论文标准训练集)
# ============================================================

class GOT10kDataset(BaseTrackingDataset):
    """
    GOT-10k 数据集加载器。

    目录结构:
        GOT-10k/
        ├── train/
        │   ├── GOT-10k_Train_000001/
        │   │   ├── 00000001.jpg
        │   │   ├── 00000002.jpg ...
        │   │   └── groundtruth.txt
        │   ├── GOT-10k_Train_000002/
        │   └── ...  (~9,335 序列)
        └── list.txt  (可选)

    标注格式: 每行 x,y,w,h (逗号分隔)
    """

    def __init__(self, root: str):
        super().__init__(root, name="GOT-10k")

    def _load_sequences(self) -> List[SequenceInfo]:
        sequences = []

        # 训练集在 train/ 子目录下，也支持直接指向 train/
        train_dir = self.root / "train"
        if not train_dir.exists():
            train_dir = self.root

        # 尝试从 list.txt 读取序列名
        list_file = train_dir / "list.txt"
        if list_file.exists():
            with open(list_file, "r") as f:
                seq_names = [l.strip() for l in f if l.strip()]
            seq_dirs = [train_dir / n for n in seq_names]
        else:
            seq_dirs = sorted([
                d for d in train_dir.iterdir()
                if d.is_dir() and (d / "groundtruth.txt").exists()
            ])

        for seq_dir in seq_dirs:
            gt_file = seq_dir / "groundtruth.txt"
            if not gt_file.exists():
                continue

            gt = self._parse_csv_gt(gt_file)
            if gt is None or len(gt) < 2:
                continue

            frame_paths = sorted(str(p) for p in seq_dir.glob("*.jpg"))
            if not frame_paths:
                frame_paths = sorted(str(p) for p in seq_dir.glob("*.png"))
            if len(frame_paths) < 2:
                continue

            # 读取 absence.label，屏蔽目标消失的帧
            absence_file = seq_dir / "absence.label"
            if absence_file.exists():
                try:
                    absence = np.loadtxt(str(absence_file))
                    if len(absence) == len(gt):
                        # absence 为 1 表示目标不在画面中，将其 gt 置为 0
                        gt[absence == 1] = [0.0, 0.0, 0.0, 0.0]
                except Exception as e:
                    logger.debug(f"读取 absence.label 失败: {seq_dir} — {e}")

            n = min(len(frame_paths), len(gt))
            sequences.append(SequenceInfo(
                name=seq_dir.name,
                frame_paths=frame_paths[:n],
                ground_truth=gt[:n],
            ))

        return sequences

    @staticmethod
    def _parse_csv_gt(gt_file: Path) -> Optional[np.ndarray]:
        """解析逗号分隔的 x,y,w,h 标注文件。"""
        try:
            gt_list = []
            with open(gt_file, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split(",")
                    if len(parts) >= 4:
                        gt_list.append([float(x) for x in parts[:4]])
            return np.array(gt_list, dtype=np.float64) if gt_list else None
        except Exception as e:
            logger.warning(f"解析 GT 失败: {gt_file} — {e}")
            return None


# ============================================================
#  LaSOT 数据集
# ============================================================

class LaSOTDataset(BaseTrackingDataset):
    """
    LaSOT 数据集加载器。

    目录结构:
        LaSOT/
        ├── airplane/
        │   ├── airplane-1/
        │   │   ├── img/
        │   │   │   ├── 00000001.jpg ...
        │   │   └── groundtruth.txt
        │   ├── airplane-2/ ...
        ├── basketball/ ...
        └── ... (70 类, ~1,400 序列)

    标注格式: 每行 x,y,w,h (逗号分隔)
    """

    def __init__(self, root: str):
        super().__init__(root, name="LaSOT")

    def _load_sequences(self) -> List[SequenceInfo]:
        sequences = []
        for cat_dir in sorted(self.root.iterdir()):
            if not cat_dir.is_dir():
                continue
            for seq_dir in sorted(cat_dir.iterdir()):
                if not seq_dir.is_dir():
                    continue

                gt_file = seq_dir / "groundtruth.txt"
                img_dir = seq_dir / "img"
                if not gt_file.exists() or not img_dir.exists():
                    continue

                gt = GOT10kDataset._parse_csv_gt(gt_file)
                if gt is None or len(gt) < 2:
                    continue

                frame_paths = sorted(str(p) for p in img_dir.glob("*.jpg"))
                if len(frame_paths) < 2:
                    continue

                n = min(len(frame_paths), len(gt))
                sequences.append(SequenceInfo(
                    name=seq_dir.name,
                    frame_paths=frame_paths[:n],
                    ground_truth=gt[:n],
                ))
        return sequences


# ============================================================
#  数据集工厂函数
# ============================================================

def build_dataset(cfg: dict, dataset_name: str) -> BaseTrackingDataset:
    """
    根据配置和数据集名称构建数据集实例。

    参数:
        cfg (dict):          完整配置字典
        dataset_name (str):  数据集名称
    返回:
        dataset: BaseTrackingDataset 实例
    """
    datasets_cfg = cfg.get("datasets", {})

    if dataset_name not in datasets_cfg:
        raise ValueError(
            f"未知数据集: '{dataset_name}'，"
            f"可用: {list(datasets_cfg.keys())}"
        )

    ds_cfg = datasets_cfg[dataset_name]
    root = ds_cfg["root"]
    ds_type = ds_cfg.get("type", dataset_name)

    DATASET_CLASSES = {
        "otb": OTB100Dataset,
        "otb100": OTB100Dataset,
        "uav": UAV123Dataset,
        "uav123": UAV123Dataset,
        "vot": VOT2022Dataset,
        "vot2022": VOT2022Dataset,
        "got10k": GOT10kDataset,
        "got10k_val": GOT10kDataset,
        "got10k_test": GOT10kDataset,
        "got": GOT10kDataset,
        "lasot": LaSOTDataset,
    }

    cls = DATASET_CLASSES.get(ds_type.lower())
    if cls is None:
        raise ValueError(
            f"不支持的数据集类型: '{ds_type}'，"
            f"可选: {list(DATASET_CLASSES.keys())}"
        )

    return cls(root)