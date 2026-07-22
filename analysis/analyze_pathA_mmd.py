
import argparse
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import torch
from scipy.stats import wilcoxon
from sklearn.preprocessing import StandardScaler

_SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPTS))
sys.path.insert(0, str(_SCRIPTS.parent))
from analyze_pathA_crosssubject import (
    load_rebrain, build_subject, SUBJECTS,
)


def dprime(F, y):
    Fi, Fo = F[y == 1], F[y == 0]
    mi, mo = Fi.mean(0), Fo.mean(0)
    d_class = np.linalg.norm(mi - mo)
    vi = ((Fi - mi) ** 2).sum(1).mean()
    vo = ((Fo - mo) ** 2).sum(1).mean()
    return float(d_class / (np.sqrt(0.5 * (vi + vo)) + 1e-12))


def _rbf(X, Y, gamma):
    d2 = (X * X).sum(1)[:, None] + (Y * Y).sum(1)[None, :] - 2.0 * X @ Y.T
    np.maximum(d2, 0, out=d2)
    return np.exp(-gamma * d2)


def mmd2(X, Y, rng, cap=250):
    if len(X) > cap:
        X = X[rng.choice(len(X), cap, replace=False)]
    if len(Y) > cap:
        Y = Y[rng.choice(len(Y), cap, replace=False)]
    Z = np.vstack([X, Y])
    dz = (Z * Z).sum(1)[:, None] + (Z * Z).sum(1)[None, :] - 2.0 * Z @ Z.T
    np.maximum(dz, 0, out=dz)
    iu = np.triu_indices(len(Z), k=1)
    sig2 = np.median(dz[iu]) + 1e-12
    gamma = 1.0 / sig2
    Kxx, Kyy, Kxy = _rbf(X, X, gamma), _rbf(Y, Y, gamma), _rbf(X, Y, gamma)
    n, m = len(X), len(Y)
    sxx = (Kxx.sum() - np.trace(Kxx)) / (n * (n - 1))
    syy = (Kyy.sum() - np.trace(Kyy)) / (m * (m - 1))
    sxy = Kxy.mean()
    return float(max(sxx + syy - 2 * sxy, 0.0))


def cross_recording_mmd(F, y, rid, cls, rng, min_n=10):
    recs = [r for r in np.unique(rid) if int(((rid == r) & (y == cls)).sum()) >= min_n]
    if len(recs) < 2:
        return None
    vals = [mmd2(F[(rid == a) & (y == cls)], F[(rid == b) & (y == cls)], rng)
            for a, b in combinations(recs, 2)]
    return float(np.mean(vals))


def run(args):
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    print(f"device={device} | out={out_dir}", flush=True)

    print("loading ReBrain...", flush=True)
    model = load_rebrain(args.checkpoint, device)

    per_sub = {}
    for sub in args.subjects:
        eeg_dir = Path(args.bids_dir) / f"sub-{sub}" / "ses-01" / "eeg"
        if not eeg_dir.exists():
            continue
        print(f"\n=== sub-{sub} ===", flush=True)
        built = build_subject(eeg_dir, model, device, args.highpass_hz,
                              args.batch_size, args.cap, args.seed)
        if built is None:
            print("  skip", flush=True)
            continue
        feats, y, rid = built
        scaler = StandardScaler().fit(np.vstack([feats["raw"], feats["hpbc"]]))
        rng = np.random.default_rng(args.seed)
        rec = {}
        for c in ["raw", "hpbc"]:
            Fz = scaler.transform(feats[c])
            rec[c] = dict(
                dprime=dprime(Fz, y),
                mmd_ictal=cross_recording_mmd(Fz, y, rid, 1, rng),
                mmd_inter=cross_recording_mmd(Fz, y, rid, 0, rng),
            )
        per_sub[sub] = rec
        _f = lambda x, n=4: "NA" if x is None else f"{x:.{n}f}"
        print(f"  dprime {_f(rec['raw']['dprime'],3)}->{_f(rec['hpbc']['dprime'],3)} | "
              f"MMD_ictal {_f(rec['raw']['mmd_ictal'])}->{_f(rec['hpbc']['mmd_ictal'])} | "
              f"MMD_inter {_f(rec['raw']['mmd_inter'])}->{_f(rec['hpbc']['mmd_inter'])}", flush=True)

    aggregate(per_sub, out_dir)


def _paired(per_sub, key):
    subs = [s for s in per_sub
            if per_sub[s]["raw"][key] is not None and per_sub[s]["hpbc"][key] is not None]
    r = np.array([per_sub[s]["raw"][key] for s in subs])
    h = np.array([per_sub[s]["hpbc"][key] for s in subs])
    return subs, r, h


def aggregate(per_sub, out_dir):
    summary = {"n_subjects": len(per_sub), "per_subject": per_sub, "paired": {}}
    for key, better in [("dprime", "up"), ("mmd_ictal", "down"), ("mmd_inter", "down")]:
        subs, r, h = _paired(per_sub, key)
        if len(r) < 3:
            continue
        try:
            _, p = wilcoxon(h, r)
        except ValueError:
            p = float("nan")
        n_good = int((h > r).sum()) if better == "up" else int((h < r).sum())
        summary["paired"][key] = dict(n=len(r), raw_mean=float(r.mean()),
                                      hpbc_mean=float(h.mean()),
                                      delta_mean=float((h - r).mean()),
                                      n_better=n_good, better=better, wilcoxon_p=float(p))
        print(f"[{key}] n={len(r)} {r.mean():.4f}->{h.mean():.4f} "
              f"(delta {float((h-r).mean()):+.4f}, {n_good}/{len(r)} better, "
              f"p={p:.4g})", flush=True)

    with open(out_dir / "mmd_metrics.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"saved -> {out_dir/'mmd_metrics.json'}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bids_dir", default="/path/to/data/chbmit_bids/BIDS_CHB-MIT")
    p.add_argument("--subjects", nargs="+", default=SUBJECTS)
    p.add_argument("--checkpoint", default="checkpoints/ieeg_gan/last.ckpt")
    p.add_argument("--highpass_hz", type=float, default=2.0)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--cap", type=int, default=800)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output_dir", default="results/figures/mmd_crosssubject")
    args = p.parse_args()
    np.random.seed(args.seed); torch.manual_seed(args.seed)
    run(args)


if __name__ == "__main__":
    main()
