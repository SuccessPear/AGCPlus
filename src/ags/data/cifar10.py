# src/ags/data/cifar.py
from torchvision import datasets, transforms
from .base import BaseDataModule

class CIFAR10DataModule(BaseDataModule):
    def setup(self):
        tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        self.train_ds = datasets.CIFAR10("data/processed/cifar10", train=True, download=True, transform=tf)
        self.val_ds   = datasets.CIFAR10("data/processed/cifar10", train=False, download=True, transform=tf)

    def train_dataset(self):
        return self.train_ds

    def val_dataset(self):
        return self.val_ds
