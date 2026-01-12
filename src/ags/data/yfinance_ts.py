import yfinance as yf
import numpy as np
import torch
from torch.utils.data import Dataset, Subset
from .base import BaseDataModule


class TimeSeriesDataset(Dataset):
    """
    Sliding-window time series dataset
    """
    def __init__(self, series, seq_len=20, target_shift=1):
        """
        series: np.ndarray, shape (T, F)
        seq_len: input sequence length
        target_shift: predict t+shift
        """
        self.series = series
        self.seq_len = seq_len
        self.target_shift = target_shift

    def __len__(self):
        return len(self.series) - self.seq_len - self.target_shift + 1

    def __getitem__(self, idx):
        x = self.series[idx : idx + self.seq_len]
        y = self.series[idx + self.seq_len + self.target_shift - 1]

        return (
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32),
        )


class YFinanceDataModule(BaseDataModule):
    def setup(self):
        """
        cfg must contain:
          - ticker
          - start
          - end
          - seq_len
          - num_data (optional)
        """
        df = yf.download(
            self.cfg.ticker,
            start=self.cfg.start,
            end=self.cfg.end,
            progress=False,
        )

        # Use OHLCV
        data = df[["Open", "High", "Low", "Close", "Volume"]].values

        # Normalize (simple, stable)
        mean = data.mean(axis=0)
        std = data.std(axis=0) + 1e-8
        data = (data - mean) / std

        full_ds = TimeSeriesDataset(
            data,
            seq_len=self.cfg.seq_len,
            target_shift=1,
        )

        # Optional subsampling (like CIFAR num_data)
        if getattr(self.cfg, "num_data", None) is not None:
            max_len = min(len(full_ds), self.cfg.num_data)
            full_ds = Subset(full_ds, list(range(max_len)))

        # Temporal split (NO stratification)
        n = len(full_ds)
        split = int(0.8 * n)

        self.train_ds = Subset(full_ds, range(0, split))
        self.val_ds = Subset(full_ds, range(split, n))
        self.test_ds = self.val_ds  # common practice in TS

    def train_dataset(self):
        return self.train_ds

    def val_dataset(self):
        return self.val_ds

    def test_dataset(self):
        return self.test_ds
