# src/ags/data/sequential_mnist.py
import torch
from torchvision import datasets, transforms
from torch.utils.data import Subset

from .base import BaseDataModule
from src.ags.utils.seed import stratified_split_indices


class _FlattenToSequence:
    """
    Convert MNIST tensor image (1, 28, 28) -> (784, 1) sequence.
    """

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(-1)          # (784,)
        x = x.unsqueeze(-1)     # (784, 1)
        return x


class SequentialMNISTDataModule(BaseDataModule):
    def setup(self):
        tf = transforms.Compose(
            [
                transforms.ToTensor(),
                _FlattenToSequence(),
            ]
        )

        trainval_for_train = datasets.MNIST(
            self.cfg.path, train=True, download=True, transform=tf
        )
        trainval_for_val = datasets.MNIST(
            self.cfg.path, train=True, download=False, transform=tf
        )
        self.test_ds = datasets.MNIST(
            self.cfg.path, train=False, download=True, transform=tf
        )

        labels = getattr(trainval_for_train, "targets", None)
        if labels is None:
            raise RuntimeError("SequentialMNISTDataModule: cannot find labels/targets.")

        tr_idx, va_idx = stratified_split_indices(
            labels, val_ratio=0.8, N=self.cfg.num_data
        )
        self.train_ds = Subset(trainval_for_train, tr_idx)
        self.val_ds = Subset(trainval_for_val, va_idx)

    def train_dataset(self):
        return self.train_ds

    def val_dataset(self):
        return self.val_ds

    def test_dataset(self):
        return self.test_ds
