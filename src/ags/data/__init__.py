# src/ags/data/__init__.py

from .base import BaseDataModule
from .cifar10 import CIFAR10DataModule
from .mnist import MNISTDataModule

__all__ = ["BaseDataModule", "build_dataset"]

def build_dataset(cfg):
    name = cfg.name.lower()
    if name == "cifar10":
        return CIFAR10DataModule(cfg)
    elif name == "mnist":
        return MNISTDataModule(cfg)
    else:
        raise ValueError(f"Unknown dataset: {name}")
