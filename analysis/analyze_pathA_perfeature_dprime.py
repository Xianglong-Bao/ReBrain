
import argparse
import json
import sys
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
from analyze_pathA_mmd import dprime

FEAT_NAMES = ["logLineLength", "logActivity", "Mobility", "Complexity", "Kurtosis", "ZCR"]
N_FEAT = len(FEAT_NAMES)


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
        rec = {"raw": {}, "hpbc": {}, "full": {}}
        for c in ["raw", "hpbc"]:
            Fz = scaler.transform(feats[c])
            rec["full"][c] = dprime(Fz, y)
            for k, name in enumerate(FEAT_NAMES):
                rec[c][name] = dprime(Fz[:, k::N_FEAT], y)
        per_sub[sub] = rec
        line = "  ".join(f"{n}:{rec['raw'][n]:.2f}->{rec['hpbc'][n]:.2f}" for n in FEAT_NAMES)
        print(f"  full dprime {rec['full']['raw']:.2f}->{rec['full']['hpbc']:.2f} | {line}", flush=True)

    aggregate(per_sub, out_dir)


def aggregate(per_sub, out_dir):
    subs = list(per_sub)
    summary = {"n_subjects": len(subs), "per_subject": per_sub, "per_feature": {}}

    rf = np.array([per_sub[s]["full"]["raw"] for s in subs])
    hf = np.array([per_sub[s]["full"]["hpbc"] for s in subs])
    _, pf = wilcoxon(hf, rf)
    summary["full_dprime"] = dict(raw_mean=float(rf.mean()), hpbc_mean=float(hf.mean()),
                                  n_up=int((hf > rf).sum()), n=len(subs), wilcoxon_p=float(pf))
    print(f"\n[full] dprime {rf.mean():.3f}->{hf.mean():.3f} {int((hf>rf).sum())}/{len(subs)} up p={pf:.3g}",
          flush=True)

    for name in FEAT_NAMES:
        r = np.array([per_sub[s]["raw"][name] for s in subs])
        h = np.array([per_sub[s]["hpbc"][name] for s in subs])
        try:
            _, p = wilcoxon(h, r)
        except ValueError:
            p = float("nan")
        summary["per_feature"][name] = dict(
            raw_mean=float(r.mean()), hpbc_mean=float(h.mean()),
            delta_mean=float((h - r).mean()), n_up=int((h > r).sum()),
            n=len(subs), wilcoxon_p=float(p))
        print(f"[{name:14}] dprime {r.mean():.3f}->{h.mean():.3f} "
              f"(delta {float((h-r).mean()):+.3f}, {int((h>r).sum())}/{len(subs)} up, p={p:.4g})", flush=True)

    with open(out_dir / "perfeature_dprime.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"saved -> {out_dir/'perfeature_dprime.json'}", flush=True)


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
    p.add_argument("--output_dir", default="results/figures/perfeature_dprime")
    args = p.parse_args()
    np.random.seed(args.seed); torch.manual_seed(args.seed)
    run(args)


if __name__ == "__main__":
    main()
