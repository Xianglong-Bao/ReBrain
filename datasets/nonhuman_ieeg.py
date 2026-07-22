
import json
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data._utils.collate import default_collate

from datasets import EEGBatch, EEGDataset


class _NpyWindows(Dataset):
    def __init__(self, index_json, window_samp, stride_samp, n_channels):
        meta = json.load(open(index_json))
        self.arrs = [np.load(m["file"], mmap_mode="r") for m in meta]
        self.window = window_samp
        self.n_channels = n_channels
        self.index = []
        for fi, a in enumerate(self.arrs):
            T = a.shape[1]
            if T < window_samp:
                continue
            n = (T - window_samp) // stride_samp + 1
            self.index.extend((fi, w * stride_samp) for w in range(n))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, n):
        fi, st = self.index[n]
        seg = np.asarray(self.arrs[fi][:, st:st + self.window], dtype=np.float32)
        C = seg.shape[0]
        if self.n_channels and C > self.n_channels:
            idx = sorted(random.sample(range(C), self.n_channels))
            seg = seg[idx]
        return EEGBatch(torch.from_numpy(seg), n, st, fi, 0)


class NonHumanIEEGData(EEGDataset):
    """Same interface as BIDSEEGData (segment_size/stride in ms); train only."""

    def __init__(
        self,
        index_json: str,
        batch_size: int = 16,
        segment_size: int = 4000,
        stride: int = 1000,
        n_channels: int = 36,
        num_workers: int = 8,
        sample_rate: int = 512,
    ) -> None:
        super().__init__()
        self.index_json = index_json
        self.batch_size = batch_size
        self.window_samp = int(segment_size / 1000.0 * sample_rate)
        self.stride_samp = int(stride / 1000.0 * sample_rate)
        self.n_channels = n_channels
        self.num_workers = num_workers

    def setup(self, stage: str = None) -> None:
        if stage in ("fit", None):
            self.dataset_train = _NpyWindows(
                self.index_json, self.window_samp, self.stride_samp, self.n_channels
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.dataset_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
            collate_fn=default_collate,
        )

    def val_dataloader(self):
        return None
