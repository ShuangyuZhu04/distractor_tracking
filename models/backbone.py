"""
backbone.py — 骨干特征提取网络
================================
职责：从原始图像中提取多尺度深度特征。
支持两种骨干：
  - ResNet50:      精度优先（默认，适用于离线训练与高精度推理）
  - MobileNetV3:   速度优先（轻量化方案，满足实时性需求）

设计说明：
  骨干网络为 SiamRPN++ 的"模板分支"和"搜索分支"共享权重，
  输出 layer2/layer3/layer4 三层特征用于多尺度匹配。
"""

import logging
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torchvision.models as models

logger = logging.getLogger(__name__)


# ============================================================
#  ResNet50 骨干（默认方案）
# ============================================================
class ResNet50Backbone(nn.Module):
    """
    基于 ResNet50 的骨干网络。

    使用空洞卷积 (Dilated Convolution) 保持 layer3/4 的空间分辨率:
        layer2 -> stride  8, channels  512
        layer3 -> stride  8, channels 1024  (dilated, rate=2)
        layer4 -> stride  8, channels 2048  (dilated, rate=4)

    所有输出层具有相同空间分辨率, 互相关输出天然对齐 (17×17),
    无需 F.interpolate 即可直接加权融合。

    参数:
        pretrained (bool):     是否加载 ImageNet 预训练权重
        frozen_stages (int):   冻结前 N 个 stage 的参数（0 表示不冻结）
        output_layers (list):  需要输出的特征层名称
    """

    # ResNet 各 stage 对应的层名与输出通道数
    STAGE_MAP = {
        "layer1": (1, 256),
        "layer2": (2, 512),
        "layer3": (3, 1024),
        "layer4": (4, 2048),
    }

    def __init__(
        self,
        pretrained: bool = True,
        frozen_stages: int = 2,
        output_layers: Optional[List[str]] = None,
    ):
        super().__init__()

        # --- 加载 torchvision 预训练 ResNet50 ---
        # 引入空洞卷积 (Dilated Convolution)
        # 保持 layer3 和 layer4 的 stride 均为 8，使空间分辨率与 layer2 完全相同
        resnet = models.resnet50(
            weights=models.ResNet50_Weights.DEFAULT if pretrained else None,
            replace_stride_with_dilation=[False, True, True]
        )
        logger.info(
            f"ResNet50 骨干初始化完成 | pretrained={pretrained}, "
            f"frozen_stages={frozen_stages}"
        )

        # --- 拆分为各 stage ---
        self.conv1 = resnet.conv1        # 7×7 卷积
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool     # stride=2 池化

        self.layer1 = resnet.layer1       # stage 1
        self.layer2 = resnet.layer2       # stage 2
        self.layer3 = resnet.layer3       # stage 3
        self.layer4 = resnet.layer4       # stage 4

        # --- 需要输出的特征层 ---
        self.output_layers = output_layers or ["layer2", "layer3", "layer4"]

        # --- 冻结指定 stage 的参数（减少显存占用 & 防止浅层特征被破坏）---
        self._freeze_stages(frozen_stages)

    def _freeze_stages(self, num_stages: int) -> None:
        """冻结前 num_stages 个 stage 的参数，使其在训练时不更新梯度。"""
        if num_stages >= 0:
            # 冻结 conv1 + bn1
            for param in self.conv1.parameters():
                param.requires_grad = False
            for param in self.bn1.parameters():
                param.requires_grad = False

        # 冻结对应的 layer
        for i in range(1, num_stages + 1):
            layer = getattr(self, f"layer{i}")
            for param in layer.parameters():
                param.requires_grad = False
            logger.debug(f"已冻结 layer{i} 参数")

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        前向传播，返回多尺度特征字典。

        参数:
            x: 输入图像张量, shape = (B, 3, H, W)

        返回:
            features: dict, key 为层名, value 为对应特征张量
                例: {"layer2": Tensor(B,512,H/8,W/8), "layer3": ..., "layer4": ...}
        """
        features = {}

        # stem: conv1 → bn → relu → maxpool
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))

        # 逐 stage 提取特征
        x = self.layer1(x)
        if "layer1" in self.output_layers:
            features["layer1"] = x

        x = self.layer2(x)
        if "layer2" in self.output_layers:
            features["layer2"] = x

        x = self.layer3(x)
        if "layer3" in self.output_layers:
            features["layer3"] = x

        x = self.layer4(x)
        if "layer4" in self.output_layers:
            features["layer4"] = x

        return features

    def get_out_channels(self) -> List[int]:
        """返回各输出层的通道数列表。"""
        return [self.STAGE_MAP[name][1] for name in self.output_layers]


# ============================================================
#  MobileNetV3 骨干（轻量化方案）
# ============================================================
class MobileNetV3Backbone(nn.Module):
    """
    基于 MobileNetV3-Large 的轻量化骨干网络。

    设计说明：
        当实时性需求高于精度需求时，使用 MobileNetV3 替代 ResNet50，
        通过深度可分离卷积大幅减少参数量与计算量。

    参数:
        pretrained (bool): 是否加载预训练权重
    """

    def __init__(self, pretrained: bool = True):
        super().__init__()

        mobilenet = models.mobilenet_v3_large(
            weights=models.MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
        )
        logger.info(f"MobileNetV3-Large 骨干初始化完成 | pretrained={pretrained}")

        # MobileNetV3 的特征提取部分是 features (Sequential)
        # 按特征图尺度拆分为三个阶段
        all_features = list(mobilenet.features.children())
        self.stage1 = nn.Sequential(*all_features[:7])    # 低层特征
        self.stage2 = nn.Sequential(*all_features[7:13])   # 中层特征
        self.stage3 = nn.Sequential(*all_features[13:])    # 高层特征

        self.output_layers = ["layer2", "layer3", "layer4"]

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """前向传播，返回多尺度特征字典。"""
        features = {}
        x = self.stage1(x)
        features["layer2"] = x    # 对齐 ResNet 的 layer 命名

        x = self.stage2(x)
        features["layer3"] = x

        x = self.stage3(x)
        features["layer4"] = x

        return features

    def get_out_channels(self) -> List[int]:
        """返回各输出层的通道数列表（需根据实际模型确认）。"""
        # MobileNetV3-Large 各阶段输出通道数
        return [80, 112, 960]


# ============================================================
#  骨干网络工厂函数
# ============================================================
def build_backbone(cfg: dict) -> nn.Module:
    """
    根据配置构建骨干网络。

    参数:
        cfg: 配置字典，需包含 backbone.type, backbone.pretrained 等字段

    返回:
        backbone: nn.Module 实例
    """
    backbone_cfg = cfg["backbone"]
    backbone_type = backbone_cfg["type"].lower()

    if backbone_type == "resnet50":
        backbone = ResNet50Backbone(
            pretrained=backbone_cfg.get("pretrained", True),
            frozen_stages=backbone_cfg.get("frozen_stages", 2),
            output_layers=backbone_cfg.get("output_layers", None),
        )
    elif backbone_type == "mobilenetv3":
        backbone = MobileNetV3Backbone(
            pretrained=backbone_cfg.get("pretrained", True),
        )
    else:
        raise ValueError(
            f"不支持的骨干网络类型: '{backbone_type}'，"
            f"可选: 'resnet50', 'mobilenetv3'"
        )

    logger.info(f"构建骨干网络: {backbone_type}")
    return backbone
