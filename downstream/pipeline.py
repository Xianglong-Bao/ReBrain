import argparse
import json
import sys
from math import gcd
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from pyedflib import EdfReader
from scipy.signal import butter, filtfilt, resample_poly
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score,
    f1_score, precision_score, recall_score,
    confusion_matrix, matthews_corrcoef, roc_auc_score,
)
from sklearn.utils.class_weight import compute_class_weight
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from models.rebrain import ReBrain
from models.encodec import Encodec, SEANetEncoder, SEANetDecoder
from vector_quantize_pytorch import ResidualVQ

WINDOW_SEC            = 4
STRIDE_SEC            = 1
SEIZURE_OVERLAP_RATIO = 0.5
REBRAIN_SR         = 512
CHBMIT_SR             = 256
CHUNK_SAMPLES_BC      = REBRAIN_SR * WINDOW_SEC




class InceptionBlock(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, 32, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv1d(in_channels, 32, kernel_size=2, stride=2, padding=0)
        self.conv3 = nn.Conv1d(in_channels, 32, kernel_size=8, stride=2, padding=3)

    def forward(self, x):
        x1 = F.elu(self.conv1(x))
        x2 = F.elu(self.conv2(x))
        x3 = F.elu(self.conv3(x))
        min_t = min(x1.shape[-1], x2.shape[-1], x3.shape[-1])
        return torch.cat([x1[..., :min_t], x2[..., :min_t], x3[..., :min_t]], dim=1)


class ChronoNet(nn.Module):
    def __init__(self, n_classes: int = 2, n_chans: int = 18,
                 n_time: int = 1024, gru_hidden: int = 32):
        super().__init__()
        self.inc1 = InceptionBlock(n_chans)
        self.inc2 = InceptionBlock(96)
        self.inc3 = InceptionBlock(96)

        t_after_inc  = n_time // 8
        H            = gru_hidden
        self.gru1    = nn.GRU(96, H, batch_first=True)
        self.affine1 = nn.Linear(t_after_inc, 1)
        self.gru2    = nn.GRU(H,     H, batch_first=True)
        self.gru3    = nn.GRU(H * 2, H, batch_first=True)
        self.gru4    = nn.GRU(H * 3, n_classes, batch_first=True)

    def forward(self, x):
        x = self.inc1(x)
        x = self.inc2(x)
        x = self.inc3(x)
        x = x.permute(0, 2, 1)
        o1, _ = self.gru1(x)
        x_res = o1
        o1_t  = o1.permute(0, 2, 1)
        o1_sq = F.elu(self.affine1(o1_t)).permute(0, 2, 1)
        o2, _ = self.gru2(o1_sq)
        x_res2 = o2
        cat1  = torch.cat([x_res[:, -1:, :], o2], dim=2)
        o3, _ = self.gru3(cat1)
        cat2  = torch.cat([x_res[:, -1:, :], x_res2, o3], dim=2)
        o4, _ = self.gru4(cat2)
        return o4.squeeze(1)




def build_rebrain(ratios) -> ReBrain:
    encoder = SEANetEncoder(
        n_filters=16, max_filters=256, dimension=64,
        kernel_size=3, last_kernel_size=3, causal=False,
        ratios=ratios, lstm=0, norm="none", true_skip=True,
    )
    decoder = SEANetDecoder(
        n_filters=16, max_filters=256, dimension=64,
        kernel_size=3, last_kernel_size=3, causal=False,
        ratios=ratios, lstm=0, norm="none", true_skip=True,
    )
    encodec = Encodec(encoder=encoder, decoder=decoder, sample_rate=512, segment=4, overlap=0)
    quantizer = ResidualVQ(
        dim=64, num_quantizers=4, codebook_size=256,
        decay=0.99, kmeans_init=True, kmeans_iters=800,
        rotation_trick=False, learnable_codebook=False,
        use_cosine_sim=False, channel_last=False,
        commitment_weight=1000, threshold_ema_dead_code=2,
    )
    return ReBrain(
        model=encodec, quantizer=quantizer,
        sample_rate=512, train_discriminator=False, use_balancer=False,
    )


def load_rebrain(ckpt_path: str, device: torch.device, ratios) -> ReBrain:
    model = build_rebrain(ratios)
    ckpt  = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.eval()
    model.to(device)
    return model


def resample(signals: np.ndarray, from_sr: int, to_sr: int) -> np.ndarray:
    if from_sr == to_sr:
        return signals
    g = gcd(to_sr, from_sr)
    return resample_poly(signals, to_sr // g, from_sr // g, axis=1).astype(np.float32)


@torch.no_grad()
def rebrain_denoise(model: ReBrain, signals_256: np.ndarray,
                       device: torch.device, batch_size: int = 8) -> np.ndarray:
    n_ch        = signals_256.shape[0]
    signals_512 = resample(signals_256, CHBMIT_SR, REBRAIN_SR)
    total       = signals_512.shape[1]
    pad = (CHUNK_SAMPLES_BC - total % CHUNK_SAMPLES_BC) % CHUNK_SAMPLES_BC
    if pad > 0:
        signals_512 = np.pad(signals_512, ((0, 0), (0, pad)))
    n_chunks = signals_512.shape[1] // CHUNK_SAMPLES_BC
    chunks   = signals_512.reshape(n_ch, n_chunks, CHUNK_SAMPLES_BC).transpose(1, 0, 2)
    denoised_chunks = []
    for i in range(0, n_chunks, batch_size):
        batch = torch.from_numpy(chunks[i:i + batch_size]).to(device)
        out   = model(batch)
        out   = rearrange(out, "(b c) 1 l -> b c l", b=batch.shape[0], c=n_ch)
        denoised_chunks.append(out.cpu().numpy())
    denoised_512 = np.concatenate(denoised_chunks, axis=0).transpose(1, 0, 2).reshape(n_ch, -1)
    denoised_512 = denoised_512[:, :total]
    denoised_256 = resample(denoised_512, REBRAIN_SR, CHBMIT_SR)
    n_orig = signals_256.shape[1]
    if denoised_256.shape[1] > n_orig:
        denoised_256 = denoised_256[:, :n_orig]
    elif denoised_256.shape[1] < n_orig:
        denoised_256 = np.pad(denoised_256, ((0, 0), (0, n_orig - denoised_256.shape[1])))
    return denoised_256


def highpass_filter(signals: np.ndarray, sfreq: float, highpass_hz: float) -> np.ndarray:
    if highpass_hz <= 0:
        return signals
    nyq = sfreq / 2.0
    b, a = butter(4, highpass_hz / nyq, btype="high")
    return filtfilt(b, a, signals, axis=1).astype(np.float32)




def read_edf(edf_path: str):
    with EdfReader(edf_path) as f:
        n_ch = f.signals_in_file
        sfreq = f.getSampleFrequency(0)
        n_samples = f.getNSamples()[0]
        signals = np.empty((n_ch, n_samples), dtype=np.float32)
        for i in range(n_ch):
            signals[i] = f.readSignal(i)
    return signals, float(sfreq)


def load_events(events_path: str):
    df = pd.read_csv(events_path, sep="\t")
    return [(float(r.onset), float(r.duration), str(r.eventType)) for _, r in df.iterrows()]


def segment_recording(signals, sfreq, events, window_sec, stride_sec, sz_overlap_ratio):
    n_ch, total_samples = signals.shape
    window_samples = int(window_sec * sfreq)
    stride_samples = int(stride_sec * sfreq)
    seizure_intervals = [(o, o + d) for o, d, t in events if t == "sz"]
    segments, labels = [], []
    for start in range(0, total_samples - window_samples + 1, stride_samples):
        end = start + window_samples
        t_start, t_end = start / sfreq, end / sfreq
        is_seizure = any(
            max(0.0, min(t_end, se) - max(t_start, ss)) / window_sec >= sz_overlap_ratio
            for ss, se in seizure_intervals
        )
        segments.append(signals[:, start:end])
        labels.append(1 if is_seizure else 0)
    if not segments:
        return np.empty((0, n_ch, window_samples), dtype=np.float32), np.empty(0, dtype=np.int64)
    return np.stack(segments).astype(np.float32), np.array(labels, dtype=np.int64)


def load_subject_data(bids_dir, subject, bc_model, device, highpass_hz=2.0, batch_size=8):
    sub_eeg_dir = Path(bids_dir) / f"sub-{subject}" / "ses-01" / "eeg"
    all_segs, all_labels, all_run_ids = [], [], []
    for edf_path in sorted(sub_eeg_dir.glob("*_eeg.edf")):
        stem = edf_path.stem.replace("_eeg", "")
        ev_path = edf_path.parent / f"{stem}_events.tsv"
        if not ev_path.exists():
            continue
        run_id = int(stem.split("run-")[1])
        events = load_events(str(ev_path))
        signals, sfreq = read_edf(str(edf_path))
        signals = highpass_filter(signals, sfreq, highpass_hz)
        signals = rebrain_denoise(bc_model, signals, device, batch_size)
        segs, labs = segment_recording(signals, sfreq, events, WINDOW_SEC, STRIDE_SEC, SEIZURE_OVERLAP_RATIO)
        if len(segs) == 0:
            continue
        all_segs.append(segs)
        all_labels.append(labs)
        all_run_ids.append(np.full(len(labs), run_id, dtype=np.int64))
    if not all_segs:
        return np.empty((0,)), np.empty((0,)), np.empty((0,))
    return np.concatenate(all_segs), np.concatenate(all_labels), np.concatenate(all_run_ids)


def normalize_segments(X):
    m = X.mean(axis=(1, 2), keepdims=True)
    s = X.std(axis=(1, 2), keepdims=True) + 1e-8
    return (X - m) / s


def downsample_background(X, y, max_ratio=5):
    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]
    if len(pos) == 0 or len(neg) == 0:
        return X, y
    max_neg = len(pos) * max_ratio
    if len(neg) > max_neg:
        neg = np.random.choice(neg, max_neg, replace=False)
    idx = np.concatenate([pos, neg])
    np.random.shuffle(idx)
    return X[idx], y[idx]




def train_and_eval(X_train, y_train, X_test, y_test, n_chans, device,
                   lr=1e-3, epochs=50, batch_size=64, patience=10, gru_hidden=32):
    X_tr, y_tr = downsample_background(X_train, y_train)
    X_tr = normalize_segments(X_tr)
    X_te = normalize_segments(X_test)

    X_tr_t = torch.from_numpy(X_tr)
    y_tr_t = torch.from_numpy(y_tr)
    X_te_t = torch.from_numpy(X_te)
    y_te_t = torch.from_numpy(y_test)

    n_time  = X_tr_t.shape[-1]
    loader  = DataLoader(TensorDataset(X_tr_t, y_tr_t), batch_size=batch_size, shuffle=True)
    model   = ChronoNet(n_classes=2, n_chans=n_chans, n_time=n_time,
                        gru_hidden=gru_hidden).float().to(device)

    weights   = compute_class_weight("balanced", classes=np.array([0, 1]), y=y_tr)
    criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor(weights).to(device))
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    best_f1, best_state, wait = -1.0, None, 0
    for epoch in range(epochs):
        model.train()
        losses = []
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            preds = model(X_te_t.to(device)).argmax(1).cpu().numpy()
        f1 = f1_score(y_te_t.numpy(), preds, zero_division=0)

        if f1 > best_f1:
            best_f1    = f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1

        if epoch % 10 == 0:
            print(f"    Epoch {epoch:3d}: loss={np.mean(losses):.4f}, F1={f1:.4f}")
        if wait >= patience:
            print(f"    Early stop at epoch {epoch}")
            break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        logits = model(X_te_t.to(device))
        probs  = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        preds  = logits.argmax(1).cpu().numpy()
    y_true = y_te_t.numpy()
    cm = confusion_matrix(y_true, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)
    try:
        auroc = float(roc_auc_score(y_true, probs))
    except ValueError:
        auroc = float('nan')
    return {
        "f1":                float(f1_score(y_true, preds, zero_division=0)),
        "precision":         float(precision_score(y_true, preds, zero_division=0)),
        "recall":            float(recall_score(y_true, preds, zero_division=0)),
        "specificity":       float(recall_score(y_true, preds, pos_label=0, zero_division=0)),
        "accuracy":          float(accuracy_score(y_true, preds)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, preds)),
        "mcc":               float(matthews_corrcoef(y_true, preds)),
        "auroc":             auroc,

        "f1_macro":          float(f1_score(y_true, preds, average='macro', zero_division=0)),
        "precision_macro":   float(precision_score(y_true, preds, average='macro', zero_division=0)),
        "f1_weighted":       float(f1_score(y_true, preds, average='weighted', zero_division=0)),
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
        "confusion_matrix":  cm.tolist(),
        "test_run": -1,
    }


def run_loso_cv(segments, labels, run_ids, n_chans, device, gru_hidden=32):
    seizure_runs = np.unique(run_ids[labels == 1])
    if len(seizure_runs) == 0:
        print("  No seizure runs found, skipping")
        return []
    results = []
    for test_run in seizure_runs:
        test_mask  = run_ids == test_run
        train_mask = ~test_mask
        X_tr, y_tr = segments[train_mask], labels[train_mask]
        X_te, y_te = segments[test_mask],  labels[test_mask]
        n_sz_te = int((y_te == 1).sum())
        print(f"  Fold run-{test_run:02d}: train={len(y_tr)}(sz={(y_tr==1).sum()}), test={len(y_te)}(sz={n_sz_te})")
        if n_sz_te == 0:
            print("    No seizure in test, skipping fold")
            continue
        res = train_and_eval(X_tr, y_tr, X_te, y_te, n_chans, device, gru_hidden=gru_hidden)
        res["test_run"] = int(test_run)
        print(f"    → F1={res['f1']:.4f}  Prec={res['precision']:.4f}  Recall={res['recall']:.4f}")
        results.append(res)
    return results




def main():
    parser = argparse.ArgumentParser(description="ChronoNet + HP + ReBrain reprogramming, CHB-MIT seizure detection")
    parser.add_argument("--bids_dir",    required=True)
    parser.add_argument("--checkpoint",  required=True)
    parser.add_argument("--ratios",      type=int, nargs="*", required=True)
    parser.add_argument("--highpass_hz", type=float, default=2.0)
    parser.add_argument("--subjects",    nargs="+",
                        default=["01","02","03","04","05","06","07","08","09","10",
                                 "11","12","13","14","15","16","17","18","19","20",
                                 "22","23","24"])
    parser.add_argument("--output_dir",  required=True)
    parser.add_argument("--gpu",         type=int, default=1)
    parser.add_argument("--batch_size",  type=int, default=8)
    parser.add_argument("--gru_hidden",  type=int, default=32)
    parser.add_argument("--seed",        type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    cr     = 2 ** len(args.ratios)
    print(f"Device: {device}")
    print(f"Model: ChronoNet  GRU hidden={args.gru_hidden}")
    print(f"Ratios: {args.ratios}  (compression: {cr}x)")
    print(f"Highpass: {args.highpass_hz} Hz  |  Checkpoint: {args.checkpoint}")

    print("Loading GAN ReBrain...")
    bc_model = load_rebrain(args.checkpoint, device, args.ratios)
    print("ReBrain loaded OK\n")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    all_results = {}

    for subject in args.subjects:
        print(f"\n{'='*60}")
        print(f"Subject sub-{subject}")
        print(f"{'='*60}")
        segments, labels, run_ids = load_subject_data(
            args.bids_dir, subject, bc_model, device,
            highpass_hz=args.highpass_hz, batch_size=args.batch_size,
        )
        if len(segments) == 0:
            print("  No data loaded, skipping")
            continue
        n_chans = segments.shape[1]
        fold_results = run_loso_cv(segments, labels, run_ids, n_chans, device,
                                   gru_hidden=args.gru_hidden)
        if fold_results:
            def _mean(k):
                vals = [r[k] for r in fold_results if k in r and not (isinstance(r[k], float) and np.isnan(r[k]))]
                return float(np.mean(vals)) if vals else float('nan')
            all_results[subject] = {
                "mean_f1":                _mean("f1"),
                "mean_precision":         _mean("precision"),
                "mean_recall":            _mean("recall"),
                "mean_specificity":       _mean("specificity"),
                "mean_accuracy":          _mean("accuracy"),
                "mean_balanced_accuracy": _mean("balanced_accuracy"),
                "mean_mcc":               _mean("mcc"),
                "mean_auroc":             _mean("auroc"),
                "mean_f1_macro":          _mean("f1_macro"),
                "mean_precision_macro":   _mean("precision_macro"),
                "mean_f1_weighted":       _mean("f1_weighted"),
                "n_folds": len(fold_results),
                "per_fold": fold_results,
            }
            r = all_results[subject]
            print(f"\n  sub-{subject} summary ({len(fold_results)} folds):")
            print(f"    Acc={r['mean_accuracy']:.4f}  F1={r['mean_f1']:.4f}  "
                  f"Prec={r['mean_precision']:.4f}  Sens={r['mean_recall']:.4f}  "
                  f"Spec={r['mean_specificity']:.4f}  BalAcc={r['mean_balanced_accuracy']:.4f}  "
                  f"MCC={r['mean_mcc']:.4f}  AUROC={r['mean_auroc']:.4f}")

    overall_f1 = overall_prec = overall_sens = overall_spec = overall_bacc = overall_mcc = overall_acc = overall_auroc = None
    overall_f1_macro = overall_prec_macro = overall_f1_weighted = None
    if all_results:
        def _omean(k):
            vals = [r[k] for r in all_results.values() if k in r and not (isinstance(r[k], float) and np.isnan(r[k]))]
            return float(np.mean(vals)) if vals else float('nan')
        overall_acc           = _omean("mean_accuracy")
        overall_f1            = _omean("mean_f1")
        overall_prec          = _omean("mean_precision")
        overall_sens          = _omean("mean_recall")
        overall_spec          = _omean("mean_specificity")
        overall_bacc          = _omean("mean_balanced_accuracy")
        overall_mcc           = _omean("mean_mcc")
        overall_auroc         = _omean("mean_auroc")
        overall_f1_macro      = _omean("mean_f1_macro")
        overall_prec_macro    = _omean("mean_precision_macro")
        overall_f1_weighted   = _omean("mean_f1_weighted")

    print(f"\n{'='*60}")
    print(f"OVERALL SUMMARY  [ChronoNet  HP {args.highpass_hz}Hz + ReBrain {cr}x]")
    print(f"{'='*60}")
    for sub, res in all_results.items():
        print(f"  sub-{sub}: Acc={res['mean_accuracy']:.4f}  F1={res['mean_f1']:.4f}  "
              f"Prec={res['mean_precision']:.4f}  Sens={res['mean_recall']:.4f}  "
              f"Spec={res['mean_specificity']:.4f}  BalAcc={res['mean_balanced_accuracy']:.4f}  "
              f"MCC={res['mean_mcc']:.4f}  AUROC={res['mean_auroc']:.4f}")
    if overall_f1 is not None:
        print(f"\n  Overall: Acc={overall_acc:.4f}  F1={overall_f1:.4f}  "
              f"Prec={overall_prec:.4f}  Sens={overall_sens:.4f}  "
              f"Spec={overall_spec:.4f}  BalAcc={overall_bacc:.4f}  "
              f"MCC={overall_mcc:.4f}  AUROC={overall_auroc:.4f}")

    out_file = out_dir / "seizure_detection_hp_bc_chrononet.json"
    with open(out_file, "w") as f:
        json.dump({
            "model":   "ChronoNet",
            "mode":    f"hp{args.highpass_hz}hz_bc{cr}x",
            "highpass_hz": args.highpass_hz,
            "ratios":  args.ratios,
            "compression_ratio": cr,
            "checkpoint": args.checkpoint,
            "subjects": args.subjects,
            "chrononet_params": {"gru_hidden": args.gru_hidden},
            "overall_mean_accuracy":          overall_acc,
            "overall_mean_f1":                overall_f1,
            "overall_mean_precision":         overall_prec,
            "overall_mean_sensitivity":       overall_sens,
            "overall_mean_specificity":       overall_spec,
            "overall_mean_balanced_accuracy": overall_bacc,
            "overall_mean_mcc":               overall_mcc,
            "overall_mean_auroc":             overall_auroc,
            "overall_mean_f1_macro":          overall_f1_macro,
            "overall_mean_precision_macro":   overall_prec_macro,
            "overall_mean_f1_weighted":       overall_f1_weighted,
            "results": all_results,
        }, f, indent=2)
    print(f"\nResults saved → {out_file}")


if __name__ == "__main__":
    main()
