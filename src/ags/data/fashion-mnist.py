# src/ags/data/mnist.py
from torchvision import datasets, transforms
from .base import BaseDataModule
from torch.utils.data import Subset
from src.ags.utils.seed import stratified_split_indices


class FashionMNISTDataModule(BaseDataModule):
    def setup(self):
        tf_train = transforms.Compose([
            transforms.ToTensor(),
        ])
        tf_eval = transforms.Compose([
            transforms.ToTensor(),
        ])

        trainval_for_train = datasets.FashionMNIST(
            self.cfg.path, train=True, download=True, transform=tf_train
        )

        self.test_ds = datasets.FashionMNIST(
            self.cfg.path, train=False, download=True, transform=tf_eval
        )

        labels = getattr(trainval_for_train, "targets", None)
        if labels is None:
            raise RuntimeError("FashionMNISTDataModule: cannot find labels/targets.")

        tr_idx, va_idx = stratified_split_indices(labels, val_ratio=0.8, N=self.cfg.num_data)
        self.train_ds = Subset(trainval_for_train, tr_idx)
        self.val_ds = Subset(trainval_for_train, va_idx)

    def train_dataset(self):
        return self.train_ds

    def val_dataset(self):
        return self.val_ds

    def test_dataset(self):
        return self.test_ds
