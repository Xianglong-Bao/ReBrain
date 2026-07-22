import argparse
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import torch
from scipy.stats import wilcoxon
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, f1_score, average_precision_score

_SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPTS))
sys.path.insert(0, str(_SCRIPTS.parent))
from analyze_representation_geometry import (
    load_rebrain, rebrain_denoise, highpass_filter,
    read_edf, load_events, window_labels,
)
from analyze_geometry_timedomain import morphology_features

SUBJECTS = ["01","02","03","04","05","06","07","08","09","10","11","12",
            "13","14","15","16","17","18","19","20","22","23","24"]


def subject_seizure_runs(eeg_dir):
    out = []
    for edf in sorted(eeg_dir.glob("*_eeg.edf")):
        stem = edf.stem.replace("_eeg", "")
        ev = edf.parent / f"{stem}_events.tsv"
        if not ev.exists():
            continue
        events = load_events(str(ev))
        if any(et == "sz" and d > 0 for _, d, et in events):
            out.append((int(stem.split("run-")[1]), edf, events))
    return out


def build_subject(eeg_dir, model, device, hp_hz, bs, cap, seed):
    rng = np.random.default_rng(seed)
    runs = subject_seizure_runs(eeg_dir)
    if len(runs) < 2:
        return None
    feats = {"raw": [], "hpbc": []}
    ys, rids, ch_ref = [], [], None
    for run_id, edf, events in runs:
        sig, sfreq, names = read_edf(str(edf))
        if ch_ref is None:
            ch_ref = names
        if len(names) != len(ch_ref):
            continue
        starts, labs, win = window_labels(sig.shape[1], sfreq, events)
        if labs.sum() == 0:
            continue
        idx_i = np.where(labs == 1)[0]
        idx_o = np.where(labs == 0)[0]
        if len(idx_o) > cap:
            idx_o = rng.choice(idx_o, cap, replace=False)
        keep = np.sort(np.concatenate([idx_i, idx_o]))
        starts_k, labs_k = starts[keep], labs[keep]
        sig_hpbc = rebrain_denoise(model, highpass_filter(sig, sfreq, hp_hz), device, bs)
        for key, s in [("raw", sig), ("hpbc", sig_hpbc)]:
            wins = np.stack([s[:, st:st + win] for st in starts_k]).astype(np.float32)
            m = morphology_features(wins)
            feats[key].append(m.reshape(m.shape[0], -1))
        ys.append(labs_k)
        rids.append(np.full(len(labs_k), run_id, dtype=np.int64))
    if len(ys) < 2:
        return None
    feats = {k: np.concatenate(v, 0) for k, v in feats.items()}
    y = np.concatenate(ys)
    rid = np.concatenate(rids)
    if len(np.unique(rid[y == 1])) < 2:
        return None
    return feats, y, rid


def geom_ratio(F, y, rid):
    sz = sorted(np.unique(rid[y == 1]).tolist())
    ic = F[y == 1].mean(0); oc = F[y == 0].mean(0)
    d_class = np.linalg.norm(ic - oc)
    ic_cens = [F[(rid == r) & (y == 1)].mean(0) for r in sz]
    drift = np.mean([np.linalg.norm(a - b) for a, b in combinations(ic_cens, 2)])
    return float(d_class), float(drift), float(d_class / (drift + 1e-12))


def lin_loso(F, y, rid):
    sz = sorted(np.unique(rid[y == 1]).tolist())
    aucs, prs, f1s = [], [], []
    for held in sz:
        tr = rid != held; te = rid == held
        if y[tr].sum() < 5 or y[te].sum() < 1:
            continue
        clf = make_pipeline(StandardScaler(),
                            LogisticRegression(max_iter=2000, class_weight="balanced"))
        clf.fit(F[tr], y[tr])
        p = clf.predict_proba(F[te])[:, 1]
        aucs.append(roc_auc_score(y[te], p))
        prs.append(average_precision_score(y[te], p))
        f1s.append(f1_score(y[te], (p >= 0.5).astype(int), zero_division=0))
    if not aucs:
        return None
    return float(np.mean(aucs)), float(np.mean(prs)), float(np.mean(f1s))


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
            print("  skip (seizure runs < 2 or channel mismatch)", flush=True)
            continue
        feats, y, rid = built
        rec = {}
        scaler = StandardScaler().fit(np.vstack([feats["raw"], feats["hpbc"]]))
        for c in ["raw", "hpbc"]:
            Fz = scaler.transform(feats[c])
            dcl, drift, ratio = geom_ratio(Fz, y, rid)
            dec = lin_loso(feats[c], y, rid)
            rec[c] = dict(d_class=dcl, drift=drift, ratio=ratio,
                          auc=dec[0] if dec else None,
                          prauc=dec[1] if dec else None,
                          f1=dec[2] if dec else None)
        rec["n_seizure_runs"] = int(len(np.unique(rid[y == 1])))
        per_sub[sub] = rec
        print(f"  seizure_runs={rec['n_seizure_runs']} | ratio raw={rec['raw']['ratio']:.3f}->"
              f"hpbc={rec['hpbc']['ratio']:.3f} | linLOSO F1 {rec['raw']['f1']:.3f}->"
              f"{rec['hpbc']['f1']:.3f} | PR-AUC {rec['raw']['prauc']:.3f}->{rec['hpbc']['prauc']:.3f}",
              flush=True)

    aggregate(per_sub, out_dir)


def _paired(per_sub, path):
    subs = [s for s in per_sub if per_sub[s]["raw"][path] is not None
            and per_sub[s]["hpbc"][path] is not None]
    r = np.array([per_sub[s]["raw"][path] for s in subs])
    h = np.array([per_sub[s]["hpbc"][path] for s in subs])
    return subs, r, h


def aggregate(per_sub, out_dir):
    summary = {"n_subjects": len(per_sub), "per_subject": per_sub, "paired": {}}
    for path in ["ratio", "f1", "prauc", "auc"]:
        subs, r, h = _paired(per_sub, path)
        if len(r) < 3:
            continue
        try:
            stat, p = wilcoxon(h, r)
        except ValueError:
            stat, p = float("nan"), float("nan")
        summary["paired"][path] = dict(
            n=len(r), raw_mean=float(r.mean()), hpbc_mean=float(h.mean()),
            delta_mean=float((h - r).mean()), n_up=int((h > r).sum()),
            wilcoxon_p=float(p))
        print(f"[{path}] n={len(r)} raw={r.mean():.3f} -> hpbc={h.mean():.3f} "
              f"(delta={float((h-r).mean()):+.3f}, {int((h>r).sum())}/{len(r)} up, "
              f"Wilcoxon p={p:.4g})", flush=True)

    with open(out_dir / "pathA_metrics.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"saved -> {out_dir/'pathA_metrics.json'}", flush=True)


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
    p.add_argument("--output_dir", default="results/figures/pathA_crosssubject_chbmit")
    args = p.parse_args()
    np.random.seed(args.seed); torch.manual_seed(args.seed)
    run(args)


if __name__ == "__main__":
    main()
