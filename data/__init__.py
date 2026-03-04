"""
data 包 — 数据集加载、Anchor 生成与训练数据管线
"""

from .datasets import (
    build_dataset,
    BaseTrackingDataset,
    OTB100Dataset,
    UAV123Dataset,
    VOT2022Dataset,
    SequenceInfo,
)
from .transforms import TrackingTransform
from .anchor import AnchorTargetGenerator
from .train_dataset import TrackingTrainDataset

__all__ = [
    "build_dataset",
    "BaseTrackingDataset",
    "OTB100Dataset",
    "UAV123Dataset",
    "VOT2022Dataset",
    "SequenceInfo",
    "TrackingTransform",
    "AnchorTargetGenerator",
    "TrackingTrainDataset",
]