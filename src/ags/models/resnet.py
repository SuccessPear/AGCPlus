# src/ags/models/resnet.py
import torch.nn as nn
from torchvision import models

def _get_num_classes(cfg):
    # Ưu tiên từ cfg.model.num_classes, fallback từ dataset
    if hasattr(cfg.model, "num_classes"):
        return cfg.model.num_classes
    if hasattr(cfg.dataset, "num_classes"):
        return cfg.dataset.num_classes
    return 10  # default an toàn

@MODELS.register("resnet")
def build_resnet(cfg):
    name = cfg.model.name.lower()
    pretrained = getattr(cfg.model, "pretrained", False)
    # weights: True → dùng weights mặc định của torchvision
    weights = "IMAGENET1K_V1" if pretrained else None

    if name == "resnet18":
        m = models.resnet18(weights=weights)
    elif name == "resnet34":
        m = models.resnet34(weights=weights)
    elif name == "resnet50":
        m = models.resnet50(weights=weights)
    else:
        raise ValueError(f"Unsupported ResNet: {name}")

    num_classes = _get_num_classes(cfg)
    in_features = m.fc.in_features
    m.fc = nn.Linear(in_features, num_classes)
    return m
