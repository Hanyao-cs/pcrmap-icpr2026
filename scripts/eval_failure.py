from __future__ import annotations

import os
import sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

import argparse
import os
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score


                                           
                                                                                
DEFAULT_SCORE_COLS = [
    "pcr_fail_mean",
    "bp_fail",
    "pcr_fail_topk_ratio",
    "pcr_fail_core_periphery",
]


@dataclass
class ScoreResult:
    score: str
    direction: str
    auroc: float
    auprc: float
    auroc_ci: Tuple[float, float]
    auprc_ci: Tuple[float, float]
    n: int
    n_groups: int


def _is_constant(x: np.ndarray) -> bool:
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return True
    return float(np.nanmax(x) - np.nanmin(x)) == 0.0


def _bootstrap_ci(
    df: pd.DataFrame,
    groups: np.ndarray,
    y_all: np.ndarray,
    score_col: str,
    n_boot: int,
    seed: int,
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    rng = np.random.default_rng(seed)
    groups = np.asarray(groups)

    uniq = np.unique(groups)
    if uniq.size <= 1:
                                   
        uniq = np.arange(len(df))

    aucs = []
    aps = []
    for _ in range(int(n_boot)):
                                        
        sampled_groups = rng.choice(uniq, size=uniq.size, replace=True)
        m = np.isin(groups, sampled_groups)
        if m.sum() < 10:
            continue

        y = y_all[m]
        s = df.loc[m, score_col].to_numpy(dtype=float)

        if np.unique(y).size < 2:
            continue
        if _is_constant(s):
            continue

        aucs.append(float(roc_auc_score(y, s)))
        aps.append(float(average_precision_score(y, s)))

    if len(aucs) == 0:
        return (float("nan"), float("nan")), (float("nan"), float("nan"))

    aucs = np.asarray(aucs, dtype=float)
    aps = np.asarray(aps, dtype=float)

    auroc_ci = (float(np.quantile(aucs, 0.025)), float(np.quantile(aucs, 0.975)))
    auprc_ci = (float(np.quantile(aps, 0.025)), float(np.quantile(aps, 0.975)))
    return auroc_ci, auprc_ci


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, required=True)
    ap.add_argument("--fail_metric", type=str, default="nrmse")
    ap.add_argument("--fail_tail", type=str, default="high", choices=["high", "low"])
    ap.add_argument("--fail_quantile", type=float, default=0.10)
    ap.add_argument("--score_cols", type=str, default="", help="Comma-separated. Empty uses defaults.")
    ap.add_argument("--group_col", type=str, default="file")
    ap.add_argument("--n_boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_csv", type=str, default="")

    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    if args.fail_metric not in df.columns:
        raise ValueError(f"fail_metric '{args.fail_metric}' not found in CSV columns.")

                                                                   
    metric = df[args.fail_metric].to_numpy(dtype=float)
    m_valid = np.isfinite(metric)
    n_drop = int((~m_valid).sum())
    if n_drop > 0:
        df = df.loc[m_valid].reset_index(drop=True)
        metric = metric[m_valid]
        print(f"[WARN] Dropped {n_drop} rows with non-finite {args.fail_metric}.")

    if df.empty:
        raise RuntimeError("No valid rows left after filtering fail_metric.")

    if args.group_col not in df.columns:
        raise ValueError(f"group_col '{args.group_col}' not found in CSV columns.")

    if args.score_cols.strip():
        score_cols = [s.strip() for s in args.score_cols.split(",") if s.strip()]
    else:
        score_cols = [c for c in DEFAULT_SCORE_COLS if c in df.columns]

    if not score_cols:
        raise RuntimeError("No valid score columns found to evaluate.")

                           
    if args.fail_tail == "high":
        thr = float(np.nanquantile(metric, 1.0 - float(args.fail_quantile)))
        y_all = (metric >= thr).astype(int)
    else:
        thr = float(np.nanquantile(metric, float(args.fail_quantile)))
        y_all = (metric <= thr).astype(int)

    groups_all = df[args.group_col].to_numpy()

    print(f"[FAIL DEF] metric={args.fail_metric} tail={args.fail_tail} q={args.fail_quantile} thr={thr:.6f}")
    print(f"[BOOT] group={args.group_col} n_boot={args.n_boot} seed={args.seed}")
    print("")

    results: List[ScoreResult] = []

    for col in score_cols:
        if col not in df.columns:
            print(f"[SKIP] {col:22s} (missing)")
            continue

        s_all = df[col].to_numpy(dtype=float)

                                             
        m = np.isfinite(s_all) & np.isfinite(metric)
        s = s_all[m]
        y = y_all[m]
        g = groups_all[m]

        if s.size < 10 or np.unique(y).size < 2:
            print(f"[SKIP] {col:22s} (insufficient)")
            continue
        if _is_constant(s):
            print(f"[SKIP] {col:22s} (constant)")
            continue

                                                                  
        direction = "+"

        auroc = float(roc_auc_score(y, s))
        auprc = float(average_precision_score(y, s))

        auroc_ci, auprc_ci = _bootstrap_ci(
            df=df.loc[m].reset_index(drop=True),
            groups=g,
            y_all=y,
            score_col=col,
            n_boot=args.n_boot,
            seed=args.seed,
        )

        r = ScoreResult(
            score=col,
            direction=direction,
            auroc=auroc,
            auprc=auprc,
            auroc_ci=auroc_ci,
            auprc_ci=auprc_ci,
            n=int(s.size),
            n_groups=int(np.unique(g).size),
        )
        results.append(r)

        print(
            f"{r.score:22s} dir={r.direction}  "
            f"AUROC={r.auroc:.4f} ({r.auroc_ci[0]:.4f},{r.auroc_ci[1]:.4f})  "
            f"AUPRC={r.auprc:.4f} ({r.auprc_ci[0]:.4f},{r.auprc_ci[1]:.4f})  "
            f"n={r.n}  groups={r.n_groups}"
        )

    if args.out_csv.strip():
        out = pd.DataFrame(
            [
                {
                    "score": r.score,
                    "dir": r.direction,
                    "auroc": r.auroc,
                    "auroc_ci_low": r.auroc_ci[0],
                    "auroc_ci_high": r.auroc_ci[1],
                    "auprc": r.auprc,
                    "auprc_ci_low": r.auprc_ci[0],
                    "auprc_ci_high": r.auprc_ci[1],
                    "n": r.n,
                    "n_groups": r.n_groups,
                    "fail_metric": args.fail_metric,
                    "fail_tail": args.fail_tail,
                    "fail_quantile": float(args.fail_quantile),
                    "fail_thr": float(thr),
                    "group_col": args.group_col,
                    "n_boot": int(args.n_boot),
                    "seed": int(args.seed),
                }
                for r in results
            ]
        )
        os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
        out.to_csv(args.out_csv, index=False)
        print(f"\n[OK] Wrote: {args.out_csv}")


if __name__ == "__main__":
    main()