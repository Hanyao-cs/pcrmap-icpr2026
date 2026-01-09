from __future__ import annotations

import os
import sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

                              
                              
import os
import sys
from pathlib import Path

_THIS_FILE = Path(__file__).resolve()


def _inject_repo_root() -> None:
    env = os.environ.get("PCRMAP_ROOT", "").strip()
    if env:
        p = Path(env).expanduser().resolve()
        if p.exists():
            return
    candidates = [_THIS_FILE.parent.parent] + list(_THIS_FILE.parents)
    for p in candidates:
        if (p / "src").is_dir():
            return


_inject_repo_root()

                              
                
                              
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _pearson_r(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2:
        return float("nan")
    x = x.astype(np.float64)
    y = y.astype(np.float64)
    x = x - x.mean()
    y = y - y.mean()
    denom = np.sqrt((x * x).sum() * (y * y).sum())
    if denom == 0:
        return float("nan")
    return float((x * y).sum() / denom)


def _spearman_r(x: np.ndarray, y: np.ndarray) -> float:
    xr = pd.Series(x).rank(method="average").to_numpy(dtype=np.float64)
    yr = pd.Series(y).rank(method="average").to_numpy(dtype=np.float64)
    return _pearson_r(xr, yr)


def _x_label(x_col: str, shifted: bool, x_mul: float) -> str:
    parts = [x_col]
    if x_mul == -1.0:
        parts.append("(-1× for orientation; higher=worse)")
    elif x_mul != 1.0:
        parts.append(f"(×{x_mul:g}; higher=worse)")
    if shifted:
        parts.append("(shifted for display; ranking unchanged)")
    return " ".join(parts)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, required=True)

                                                        
    ap.add_argument("--x_col", type=str, default="", help="Score column (default picks pcr_fail_mean if present)")
    ap.add_argument("--score_col", type=str, default="", help="Alias of --x_col")

    ap.add_argument("--y_col", type=str, default="nrmse")
    ap.add_argument("--fail_quantile", type=float, default=0.10)
    ap.add_argument("--fail_tail", type=str, default="high", choices=["high", "low"])

    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--out_name", type=str, default="fig3_scatter")
    ap.add_argument("--title", type=str, default="")

    ap.add_argument("--logx", action="store_true")
    ap.add_argument("--alpha", type=float, default=0.35)
    ap.add_argument("--s", type=float, default=14.0)

    ap.add_argument(
        "--shift_fail_scores",
        action="store_true",
        help="Shift scores to be non-negative for display (does NOT change ranking/correlation)",
    )

    ap.add_argument(
        "--x_mul",
        type=float,
        default=1.0,
        help="Multiply x by this value BEFORE plotting/correlation (use -1 to flip orientation).",
    )

    ap.add_argument(
        "--group_col",
        type=str,
        default="",
        help="Optional: aggregate by a column (e.g., file). Uses mean for x/y.",
    )

    args = ap.parse_args()

    if (not args.x_col.strip()) and args.score_col.strip():
        args.x_col = args.score_col.strip()

    df = pd.read_csv(args.csv)

    if not args.x_col.strip():
        for cand in ["pcr_fail_mean", "bp_fail", "pcr_mean", "bp_rel_mean", "resid_norm"]:
            if cand in df.columns:
                args.x_col = cand
                break
        else:
            raise RuntimeError("No suitable default x_col found. Please pass --x_col explicitly.")

    if args.x_col not in df.columns or args.y_col not in df.columns:
        raise RuntimeError(f"CSV must contain columns: {args.x_col}, {args.y_col}")

                                             
    if args.group_col.strip():
        gc = args.group_col.strip()
        if gc not in df.columns:
            raise RuntimeError(f"--group_col {gc} not in CSV columns.")
        df = df.groupby(gc, as_index=False)[[args.x_col, args.y_col]].mean()

    x_raw = df[args.x_col].to_numpy(dtype=float)
    y = df[args.y_col].to_numpy(dtype=float)

                                 
    x = x_raw * float(args.x_mul)

                   
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]
    y = y[m]

                                              
    shifted = False
    x_plot = x.copy()
    if args.shift_fail_scores or ("fail" in args.x_col):
        x_min = float(np.min(x_plot)) if x_plot.size else 0.0
        if x_min <= 0:
            x_plot = x_plot - x_min
            if args.logx:
                x_plot = x_plot + 1e-9                          
            shifted = True

                             
    if args.logx:
        m2 = x_plot > 0
        x_plot = x_plot[m2]
        y = y[m2]
        x = x[m2]                                                   

                                                          
    if args.fail_tail == "high":
        thr = float(np.quantile(y, 1.0 - args.fail_quantile))
        fail = y >= thr
        fail_desc = f"failure = top {int(args.fail_quantile*100)}% {args.y_col}"
    else:
        thr = float(np.quantile(y, args.fail_quantile))
        fail = y <= thr
        fail_desc = f"failure = bottom {int(args.fail_quantile*100)}% {args.y_col}"

                                                                    
    pr = _pearson_r(x, y)
    sr = _spearman_r(x, y)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(7.0, 5.0))
    ax = plt.gca()

    ax.scatter(x_plot[~fail], y[~fail], s=args.s, alpha=args.alpha, label="non-fail")
    ax.scatter(x_plot[fail], y[fail], s=args.s, alpha=min(1.0, args.alpha + 0.25), label="fail")
    ax.axhline(thr, linestyle="--", linewidth=1.0)

    ax.set_xlabel(_x_label(args.x_col, shifted, float(args.x_mul)))
    ax.set_ylabel(args.y_col)

    if args.title.strip():
        title_main = args.title
    else:
        title_main = f"{fail_desc} | thr={thr:.3f}"
        if args.group_col.strip():
            title_main += f" | agg={args.group_col.strip()}(mean)"

    title_sub = f"Pearson r={pr:.3f}, Spearman r={sr:.3f}"
    ax.set_title(f"{title_main}\n{title_sub}")

    if args.logx:
        ax.set_xscale("log")

    ax.legend(loc="upper left", frameon=True)

    fig.tight_layout()

    png = out_dir / f"{args.out_name}.png"
    pdf = out_dir / f"{args.out_name}.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight", pad_inches=0.05)
    fig.savefig(pdf, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)

    print(f"[OK] Saved: {png}")
    print(f"[OK] Saved: {pdf}")
    print(f"[INFO] thr({args.fail_tail}, q={args.fail_quantile}) = {thr:.6f}")
    print(f"[INFO] n_points = {y.size} | n_fail = {int(fail.sum())} ({fail.mean()*100:.2f}%)")


if __name__ == "__main__":
    main()