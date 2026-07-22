

import sys
from math import gcd
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from einops import rearrange
from pyedflib import EdfReader
from scipy.signal import butter, filtfilt, resample_poly

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from models.rebrain import ReBrain
from models.encodec import Encodec, SEANetEncoder, SEANetDecoder
from vector_quantize_pytorch import ResidualVQ

WINDOW_SEC = 4
STRIDE_SEC = 1
SEIZURE_OVERLAP_RATIO = 0.5
REBRAIN_SR = 512
CHBMIT_SR = 256
CHUNK_SAMPLES_BC = REBRAIN_SR * WINDOW_SEC


def build_rebrain() -> ReBrain:
    encoder = SEANetEncoder(n_filters=16, max_filters=256, dimension=64,
                            kernel_size=3, last_kernel_size=3, causal=False,
                            ratios=[2, 2, 2, 2, 2, 2], lstm=0, norm="none", true_skip=True)
    decoder = SEANetDecoder(n_filters=16, max_filters=256, dimension=64,
                            kernel_size=3, last_kernel_size=3, causal=False,
                            ratios=[2, 2, 2, 2, 2, 2], lstm=0, norm="none", true_skip=True)
    encodec = Encodec(encoder=encoder, decoder=decoder, sample_rate=512, segment=4, overlap=0)
    quantizer = ResidualVQ(dim=64, num_quantizers=4, codebook_size=256,
                           decay=0.99, kmeans_init=True, kmeans_iters=800,
                           rotation_trick=False, learnable_codebook=False,
                           use_cosine_sim=False, channel_last=False,
                           commitment_weight=1000, threshold_ema_dead_code=2)
    return ReBrain(model=encodec, quantizer=quantizer, sample_rate=512,
                   train_discriminator=False, use_balancer=False)


def load_rebrain(ckpt_path: str, device: torch.device) -> ReBrain:
    model = build_rebrain()
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    return model.eval().to(device)


def _resample(sig: np.ndarray, from_sr: int, to_sr: int) -> np.ndarray:
    if from_sr == to_sr:
        return sig
    g = gcd(to_sr, from_sr)
    return resample_poly(sig, to_sr // g, from_sr // g, axis=1).astype(np.float32)


@torch.no_grad()
def rebrain_denoise(model: ReBrain, signals_256: np.ndarray,
                    device: torch.device, batch_size: int = 8) -> np.ndarray:
    n_ch = signals_256.shape[0]
    sig_512 = _resample(signals_256, CHBMIT_SR, REBRAIN_SR)
    total = sig_512.shape[1]
    pad = (CHUNK_SAMPLES_BC - total % CHUNK_SAMPLES_BC) % CHUNK_SAMPLES_BC
    if pad > 0:
        sig_512 = np.pad(sig_512, ((0, 0), (0, pad)))
    n_chunks = sig_512.shape[1] // CHUNK_SAMPLES_BC
    chunks = sig_512.reshape(n_ch, n_chunks, CHUNK_SAMPLES_BC).transpose(1, 0, 2)
    out = []
    for i in range(0, n_chunks, batch_size):
        batch = torch.from_numpy(chunks[i:i + batch_size]).to(device)
        o = model(batch)
        o = rearrange(o, "(b c) 1 l -> b c l", b=batch.shape[0], c=n_ch)
        out.append(o.cpu().numpy())
    den_512 = np.concatenate(out, axis=0).transpose(1, 0, 2).reshape(n_ch, -1)[:, :total]
    den_256 = _resample(den_512, REBRAIN_SR, CHBMIT_SR)
    n_orig = signals_256.shape[1]
    if den_256.shape[1] > n_orig:
        den_256 = den_256[:, :n_orig]
    elif den_256.shape[1] < n_orig:
        den_256 = np.pad(den_256, ((0, 0), (0, n_orig - den_256.shape[1])))
    return den_256


def highpass_filter(signals: np.ndarray, sfreq: float, hz: float) -> np.ndarray:
    if hz <= 0:
        return signals
    b, a = butter(4, hz / (sfreq / 2.0), btype="high")
    return filtfilt(b, a, signals, axis=1).astype(np.float32)


def read_edf(edf_path: str):
    with EdfReader(edf_path) as f:
        n_ch = f.signals_in_file
        sfreq = float(f.getSampleFrequency(0))
        n_samp = f.getNSamples()[0]
        sig = np.empty((n_ch, n_samp), dtype=np.float32)
        names = [f.getLabel(i).strip() for i in range(n_ch)]
        for i in range(n_ch):
            sig[i] = f.readSignal(i)
    return sig, sfreq, names


def load_events(events_path: str):
    df = pd.read_csv(events_path, sep="\t")
    return [(float(r.onset), float(r.duration), str(r.eventType)) for _, r in df.iterrows()]


def window_labels(total_samples, sfreq, events):
    win = int(WINDOW_SEC * sfreq)
    stride = int(STRIDE_SEC * sfreq)
    sz = [(o, o + d) for o, d, et in events if et == "sz"]
    starts, labels = [], []
    for s in range(0, total_samples - win + 1, stride):
        e = s + win
        ts, te = s / sfreq, e / sfreq
        is_sz = any(max(0.0, min(te, b) - max(ts, a)) / WINDOW_SEC >= SEIZURE_OVERLAP_RATIO
                    for a, b in sz)
        starts.append(s)
        labels.append(1 if is_sz else 0)
    return np.array(starts, dtype=np.int64), np.array(labels, dtype=np.int64), win
