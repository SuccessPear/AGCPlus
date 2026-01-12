# src/ags/data/cifar.py
from torchvision import datasets, transforms
from .base import BaseDataModule
from torch.utils.data import Subset
from src.ags.utils.seed import stratified_split_indices

class CIFAR10DataModule(BaseDataModule):
    def setup(self):
        tf_train = transforms.Compose([
            # transforms.RandomCrop(32, padding=4),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # transforms.Normalize(mean=(0.4914, 0.4822, 0.4465),
            #                      std=(0.2470, 0.2435, 0.2616)),
        ])
        tf_eval = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize(mean=(0.4914, 0.4822, 0.4465),
            #                      std=(0.2470, 0.2435, 0.2616)),
        ])
        trainval_for_train = datasets.CIFAR10(self.cfg.path, train=True, download=True, transform=tf_train)
        trainval_for_val = datasets.CIFAR10(self.cfg.path, train=True, download=False, transform=tf_eval)
        self.test_ds = datasets.CIFAR10(self.cfg.path, train=False, download=True, transform=tf_eval)

        labels = getattr(trainval_for_train, "targets", None)
        if labels is None:
            raise RuntimeError("CIFAR10DataModule: không tìm thấy labels/targets.")

        tr_idx, va_idx = stratified_split_indices(labels, val_ratio=0.8, N = self.cfg.num_data)
        self.train_ds = Subset(trainval_for_train, tr_idx)
        self.val_ds = Subset(trainval_for_val, va_idx)

    def train_dataset(self):
        return self.train_ds

    def val_dataset(self):
        return self.val_ds

    def test_dataset(self):
        return self.test_ds
