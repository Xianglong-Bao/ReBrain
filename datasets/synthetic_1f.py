
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data._utils.collate import default_collate

from datasets import EEGBatch, EEGDataset

_TWO_PI = float(2 * np.pi)


class _Synth1FWindows(Dataset):
    def __init__(self, target_amp_path, window_samp, n_channels, length, repeat_single=False):
        amp = np.load(target_amp_path).astype(np.float32)
        assert window_samp == 2 * (len(amp) - 1), \
            f"window {window_samp} does not match target_amp bins {len(amp)} (need N=2*(bins-1))"
        self.amp = torch.from_numpy(amp)
        self.N = window_samp
        self.n_channels = n_channels
        self.length = length


        self.repeat_single = repeat_single
        self._fixed = self._gen() if repeat_single else None

    def __len__(self):
        return self.length

    def _gen(self):
        nb = self.amp.shape[0]
        phases = torch.rand(self.n_channels, nb) * _TWO_PI
        real = self.amp.unsqueeze(0) * torch.cos(phases)
        imag = self.amp.unsqueeze(0) * torch.sin(phases)
        imag[:, 0] = 0.0
        imag[:, -1] = 0.0
        return torch.fft.irfft(torch.complex(real, imag), n=self.N, dim=1).float()

    def __getitem__(self, idx):
        sig = self._fixed if self.repeat_single else self._gen()
        return EEGBatch(sig, idx, 0, 0, 0)


class Synthetic1FData(EEGDataset):
    """Spectrum-matched synthetic 1/f data module (train only, same interface as the other controls)."""

    def __init__(
        self,
        target_amp_path: str,
        batch_size: int = 16,
        segment_size: int = 4000,
        stride: int = 1000,
        n_channels: int = 36,
        num_workers: int = 8,
        sample_rate: int = 512,
        length: int = 100000,
        repeat_single: bool = False,
    ) -> None:
        super().__init__()
        self.target_amp_path = target_amp_path
        self.batch_size = batch_size
        self.window_samp = int(segment_size / 1000.0 * sample_rate)
        self.n_channels = n_channels
        self.num_workers = num_workers
        self.length = length
        self.repeat_single = repeat_single

    def setup(self, stage: str = None) -> None:
        if stage in ("fit", None):
            self.dataset_train = _Synth1FWindows(
                self.target_amp_path, self.window_samp, self.n_channels,
                self.length, self.repeat_single)

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
