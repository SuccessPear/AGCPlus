# src/ags/data/base.py
import torch
from torch.utils.data import DataLoader
from abc import ABC, abstractmethod

class BaseDataModule(ABC):
    def __init__(self, cfg):
        self.cfg = cfg
        self.batch_size = cfg.trainer.batch_size
        self.num_workers = getattr(cfg.trainer, "num_workers", 4)

    @abstractmethod
    def setup(self):
        """Tải và xử lý dữ liệu (download nếu cần)."""
        pass

    @abstractmethod
    def train_dataset(self):
        """Trả về torch Dataset cho training."""
        pass

    @abstractmethod
    def val_dataset(self):
        """Trả về torch Dataset cho validation."""
        pass

    def train_loader(self):
        return DataLoader(
            self.train_dataset(),
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_loader(self):
        return DataLoader(
            self.val_dataset(),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
